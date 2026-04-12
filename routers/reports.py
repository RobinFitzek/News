"""
Weekly report preview, PDF export, report settings.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/api/report/preview", response_class=HTMLResponse)
async def api_report_preview(request: Request, username: str = Depends(require_api_key_or_session)):
    """Generate and return the weekly report as HTML (opens in browser)."""
    from engine.report_generator import ReportGenerator
    rg = ReportGenerator()
    result = rg.generate_weekly_report()
    html = result.get("html_content", "")
    if not html:
        html = "<html><body><p>Report generation failed: " + result.get("error", "unknown error") + "</p></body></html>"
    return HTMLResponse(content=html)


@router.post("/api/report/send")
async def api_report_send(request: Request, username: str = Depends(require_api_key_or_session)):
    """Generate and send the weekly report via configured channels."""
    from engine.report_generator import ReportGenerator
    rg = ReportGenerator()
    result = rg.generate_weekly_report()
    html = result.get("html_content", "")
    if not html:
        return {"sent": False, "error": result.get("error", "Generation failed")}
    sent = rg.send_report(html)
    return {"sent": sent, "generated_at": result.get("generated_at"), "error": result.get("error")}


@router.post("/settings/save-report")
async def settings_save_report(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth),
):
    """Save weekly report schedule settings."""
    csrf.verify_token(request, csrf_token)
    form = await request.form()
    db.set_setting("weekly_report_auto_send", "1" if form.get("weekly_report_auto_send") else "0")
    db.set_setting("weekly_report_day", form.get("weekly_report_day", "Sunday"))
    db.set_setting("weekly_report_time", form.get("weekly_report_time", "08:00"))
    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.get("/report/weekly/pdf")
async def report_weekly_pdf(request: Request, username: str = Depends(require_auth)):
    """Render the weekly HTML report to PDF via WeasyPrint (#36)."""
    from fastapi.responses import Response
    from engine.report_generator import ReportGenerator

    rg = ReportGenerator()
    result = rg.generate_weekly_report()
    html = result.get("html_content", "")
    if not html:
        raise HTTPException(status_code=500, detail=result.get("error", "Report generation failed"))

    try:
        from weasyprint import HTML as WeasyHTML
        pdf_bytes = WeasyHTML(string=html).write_pdf()
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="PDF export requires weasyprint. Run: pip install weasyprint",
        )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=weekly_report.pdf"},
    )
