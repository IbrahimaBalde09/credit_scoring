import csv
import io
import json
import logging
from datetime import datetime, time

from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.core.paginator import Paginator
from django.db import connection
from django.db.models import Avg, Count, Sum
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.utils.dateparse import parse_date
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

try:
    from django_ratelimit.decorators import ratelimit
except ImportError:  # pragma: no cover - garde-fou si le paquet manque
    def ratelimit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from .auth import is_valid_api_key
from .forms import CreditForm
from .models import ScoreHistory
from .services import build_result, model_available, shap_available, get_model_version

logger = logging.getLogger("scoring")
audit_logger = logging.getLogger("scoring.audit")


def get_client_ip(request) -> str | None:
    forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")


def save_history(cleaned_data, result, *, source, request=None, analyst=None):
    try:
        entry = ScoreHistory.objects.create(
            client_number=str(cleaned_data["client_number"]),
            loan_amnt=float(cleaned_data["loan_amnt"]),
            person_income=float(cleaned_data["person_income"]),
            loan_int_rate=float(cleaned_data["loan_int_rate"]),
            debt_ratio=float(cleaned_data["debt_ratio"]),
            person_age=int(cleaned_data["person_age"]),
            person_emp_length=float(cleaned_data["person_emp_length"]),
            cb_person_cred_hist_length=float(cleaned_data["cb_person_cred_hist_length"]),
            loan_grade=str(cleaned_data.get("loan_grade", "")),
            person_home_ownership=str(cleaned_data.get("person_home_ownership", "")),
            loan_intent=str(cleaned_data.get("loan_intent", "")),
            cb_person_default_on_file=str(cleaned_data.get("cb_person_default_on_file", "")),
            probability=float(result["prob"]),
            decision=result["decision"],
            risk=result["risk"],
            analyst=analyst,
            model_version=result.get("model_version", ""),
            source=source,
            request_ip=get_client_ip(request) if request is not None else None,
        )
        audit_logger.info(
            "scoring_decision client=%s decision=%s probability=%.2f source=%s "
            "analyst=%s model_version=%s ip=%s",
            entry.client_number, entry.decision, entry.probability, entry.source,
            getattr(analyst, "username", None), entry.model_version, entry.request_ip,
        )
    except Exception:
        logger.exception("Erreur lors de l'enregistrement de l'historique de scoring")
        raise


def get_filtered_queryset(request):
    decision_filter = request.GET.get("decision", "").strip()
    start_date_str = request.GET.get("start_date", "").strip()
    end_date_str = request.GET.get("end_date", "").strip()
    client_search = request.GET.get("client_search", "").strip()
    sort = request.GET.get("sort", "-created_at").strip()

    allowed_sorts = {
        "created_at": "created_at",
        "-created_at": "-created_at",
        "client_number": "client_number",
        "-client_number": "-client_number",
        "loan_amnt": "loan_amnt",
        "-loan_amnt": "-loan_amnt",
        "person_income": "person_income",
        "-person_income": "-person_income",
        "probability": "probability",
        "-probability": "-probability",
    }
    sort = allowed_sorts.get(sort, "-created_at")

    qs = ScoreHistory.objects.all()

    if decision_filter in {"ACCEPT", "REVIEW", "REJECT"}:
        qs = qs.filter(decision=decision_filter)

    if client_search:
        qs = qs.filter(client_number__icontains=client_search)

    start_date = parse_date(start_date_str) if start_date_str else None
    end_date = parse_date(end_date_str) if end_date_str else None

    if start_date:
        start_dt = datetime.combine(start_date, time.min)
        qs = qs.filter(created_at__gte=start_dt)

    if end_date:
        end_dt = datetime.combine(end_date, time.max)
        qs = qs.filter(created_at__lte=end_dt)

    return qs, decision_filter, start_date_str, end_date_str, client_search, sort


def decision_label(code):
    mapping = {
        "ACCEPT": "Accepté",
        "REVIEW": "À examiner",
        "REJECT": "Refusé",
    }
    return mapping.get(code, code)


@login_required
def home(request):
    result = None
    error_message = None

    if request.method == "POST":
        form = CreditForm(request.POST)

        if form.is_valid():
            try:
                result = build_result(form.cleaned_data)
                save_history(
                    form.cleaned_data, result,
                    source="WEB", request=request, analyst=request.user,
                )
            except FileNotFoundError as e:
                error_message = str(e)
            except Exception:
                logger.exception("Erreur pendant le scoring (formulaire web)")
                error_message = "Une erreur est survenue pendant le scoring. Merci de réessayer."
    else:
        form = CreditForm()

    return render(
        request,
        "scoring/home.html",
        {
            "form": form,
            "result": result,
            "error_message": error_message,
            "model_available": model_available(),
            "shap_available": shap_available(),
        },
    )


@login_required
def dashboard(request):
    qs, decision_filter, start_date_str, end_date_str, client_search, sort = get_filtered_queryset(request)
    page_number = request.GET.get("page", 1)

    try:
        total = qs.count()
        avg_probability = qs.aggregate(avg=Avg("probability"))["avg"] or 0

        decision_counts_qs = (
            qs.values("decision")
            .annotate(count=Count("id"))
            .order_by()
        )

        decision_counts = {
            "ACCEPT": 0,
            "REVIEW": 0,
            "REJECT": 0,
        }
        for row in decision_counts_qs:
            decision_counts[row["decision"]] = row["count"]

        accept_count = decision_counts["ACCEPT"]
        review_count = decision_counts["REVIEW"]
        reject_count = decision_counts["REJECT"]

        accept_rate = round((accept_count / total) * 100, 2) if total else 0
        review_rate = round((review_count / total) * 100, 2) if total else 0
        reject_rate = round((reject_count / total) * 100, 2) if total else 0

        low_risk = qs.filter(probability__lt=10).count()
        medium_risk = qs.filter(probability__gte=10, probability__lt=30).count()
        high_risk = qs.filter(probability__gte=30).count()

        if avg_probability < 10:
            portfolio_risk_label = "Faible"
        elif avg_probability < 30:
            portfolio_risk_label = "Modéré"
        else:
            portfolio_risk_label = "Élevé"

        exposure_total = qs.aggregate(total=Sum("loan_amnt"))["total"] or 0
        exposure_accept = qs.filter(decision="ACCEPT").aggregate(total=Sum("loan_amnt"))["total"] or 0
        exposure_review = qs.filter(decision="REVIEW").aggregate(total=Sum("loan_amnt"))["total"] or 0
        exposure_reject = qs.filter(decision="REJECT").aggregate(total=Sum("loan_amnt"))["total"] or 0

        # Perte attendue = somme(montant * probabilité de défaut), calculée
        # en base plutôt qu'en itérant en Python ligne à ligne.
        expected_loss = 0
        for loan_amnt, probability in qs.values_list("loan_amnt", "probability"):
            expected_loss += float(loan_amnt) * (float(probability) / 100.0)
        expected_loss = round(expected_loss, 2)

        trend_items = list(qs.order_by("-created_at")[:20])
        chart_labels = []
        chart_probabilities = []
        for item in reversed(trend_items):
            chart_labels.append(item.created_at.strftime("%d/%m %H:%M"))
            chart_probabilities.append(round(item.probability, 2))

        table_qs = qs.order_by(sort)
        paginator = Paginator(table_qs, 10)
        page_obj = paginator.get_page(page_number)

    except Exception:
        logger.exception("Erreur lors du calcul des indicateurs du dashboard")
        total = 0
        avg_probability = 0
        accept_count = 0
        review_count = 0
        reject_count = 0
        accept_rate = 0
        review_rate = 0
        reject_rate = 0
        low_risk = 0
        medium_risk = 0
        high_risk = 0
        portfolio_risk_label = "Non disponible"
        exposure_total = 0
        exposure_accept = 0
        exposure_review = 0
        exposure_reject = 0
        expected_loss = 0
        chart_labels = []
        chart_probabilities = []
        page_obj = []

    context = {
        "total": total,
        "avg_probability": round(avg_probability, 2),
        "accept_count": accept_count,
        "review_count": review_count,
        "reject_count": reject_count,
        "accept_rate": accept_rate,
        "review_rate": review_rate,
        "reject_rate": reject_rate,
        "page_obj": page_obj,
        "chart_labels": json.dumps(chart_labels, ensure_ascii=False),
        "chart_probabilities": json.dumps(chart_probabilities),
        "risk_bucket_labels": json.dumps(
            ["Faible (<10%)", "Modéré (10-30%)", "Élevé (>=30%)"],
            ensure_ascii=False
        ),
        "risk_bucket_values": json.dumps([low_risk, medium_risk, high_risk]),
        "portfolio_risk_label": portfolio_risk_label,
        "exposure_total": round(exposure_total, 2),
        "exposure_accept": round(exposure_accept, 2),
        "exposure_review": round(exposure_review, 2),
        "exposure_reject": round(exposure_reject, 2),
        "expected_loss": expected_loss,
        "decision_filter": decision_filter,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "client_search": client_search,
        "sort": sort,
    }
    return render(request, "scoring/dashboard.html", context)


@login_required
def export_csv(request):
    qs, _, _, _, client_search, sort = get_filtered_queryset(request)
    rows = qs.order_by(sort)

    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = 'attachment; filename="dashboard_credit_scoring.csv"'

    response.write("\ufeff")
    writer = csv.writer(response, delimiter=";")
    writer.writerow([
        "Numéro client",
        "Date",
        "Montant",
        "Revenu",
        "Taux intérêt",
        "Taux endettement",
        "Probabilité défaut (%)",
        "Décision",
        "Risque",
        "Grade",
        "Logement",
        "Objet prêt",
    ])

    for item in rows:
        writer.writerow([
            item.client_number,
            item.created_at.strftime("%d/%m/%Y %H:%M"),
            item.loan_amnt,
            item.person_income,
            item.loan_int_rate,
            item.debt_ratio,
            round(item.probability, 2),
            decision_label(item.decision),
            item.risk,
            item.loan_grade,
            item.person_home_ownership,
            item.loan_intent,
        ])

    audit_logger.info("export_csv user=%s rows=%d", request.user.username, rows.count())
    return response


@login_required
def export_pdf(request):
    qs, decision_filter, start_date_str, end_date_str, client_search, sort = get_filtered_queryset(request)
    rows = qs.order_by(sort)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1 * cm, leftMargin=1 * cm)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Export Dashboard Credit Scoring", styles["Title"]))
    elements.append(Spacer(1, 12))

    filters_text = (
        f"Décision: {decision_label(decision_filter) if decision_filter else 'Toutes'} | "
        f"Date début: {start_date_str or 'Non définie'} | "
        f"Date fin: {end_date_str or 'Non définie'} | "
        f"Recherche client: {client_search or 'Aucune'}"
    )
    elements.append(Paragraph(filters_text, styles["Normal"]))
    elements.append(Spacer(1, 12))

    table_data = [[
        "Client", "Date", "Montant", "Revenu", "Probabilité", "Décision", "Risque"
    ]]

    for item in rows[:200]:
        table_data.append([
            item.client_number,
            item.created_at.strftime("%d/%m/%Y %H:%M"),
            f"{item.loan_amnt:.2f}",
            f"{item.person_income:.2f}",
            f"{item.probability:.2f}%",
            decision_label(item.decision),
            item.risk,
        ])

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3f5e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
    ]))
    elements.append(table)

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()

    audit_logger.info("export_pdf user=%s rows=%d", request.user.username, len(table_data) - 1)

    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="dashboard_credit_scoring.pdf"'
    response.write(pdf)
    return response


@csrf_exempt
@require_http_methods(["POST"])
@ratelimit(key="header:x-api-key", rate="60/m", block=True)
def api_score(request):
    """
    Endpoint machine-à-machine pour les systèmes internes du SI bancaire.
    Authentification par clé API (`X-API-Key`), pas de session utilisateur.
    """
    if not is_valid_api_key(request):
        logger.warning("Tentative d'accès à l'API de scoring avec une clé invalide ou absente (ip=%s)", get_client_ip(request))
        return JsonResponse({"error": "Authentification invalide ou manquante (X-API-Key)."}, status=401)

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "JSON invalide."}, status=400)

    form = CreditForm(payload)

    if not form.is_valid():
        return JsonResponse(
            {
                "error": "Données invalides.",
                "details": form.errors,
            },
            status=400,
        )

    try:
        result = build_result(form.cleaned_data)
        save_history(form.cleaned_data, result, source="API", request=request, analyst=None)

        return JsonResponse(
            {
                "client_number": form.cleaned_data["client_number"],
                "probability": result["prob"],
                "decision_code": result["decision"],
                "decision": result["decision_fr"],
                "risk": result["risk"],
                "interpretation": result["interpretation"],
                "thresholds": result["thresholds"],
                "model_version": result["model_version"],
                "negative_factors": result["neg"],
                "positive_factors": result["pos"],
                "shap": result["shap"],
            }
        )
    except FileNotFoundError as e:
        return JsonResponse({"error": str(e)}, status=500)
    except Exception:
        logger.exception("Erreur pendant le scoring (API)")
        return JsonResponse({"error": "Erreur interne pendant le scoring."}, status=500)


@method_decorator(
    ratelimit(key="ip", rate="10/m", method="POST", block=True),
    name="post",
)
class RateLimitedLoginView(LoginView):
    """Login protégé contre le bruteforce (par IP)."""
    template_name = "scoring/login.html"


def healthz(request):
    """
    Endpoint de supervision (utilisé par le healthcheck Docker / le load
    balancer). Vérifie que la base de données répond et que le modèle ML
    est chargé. Ne nécessite pas d'authentification (standard pour un
    /healthz), ne renvoie aucune donnée sensible.
    """
    db_ok = True
    try:
        connection.ensure_connection()
    except Exception:
        db_ok = False

    is_model_loaded = model_available()
    status_ok = db_ok and is_model_loaded

    payload = {
        "status": "ok" if status_ok else "degraded",
        "database": db_ok,
        "model_loaded": is_model_loaded,
        "shap_loaded": shap_available(),
        "model_version": get_model_version() if is_model_loaded else None,
    }
    return JsonResponse(payload, status=200 if status_ok else 503)
