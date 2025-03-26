from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from io import BytesIO
from datetime import datetime
from collections import defaultdict
from models.models import research_results
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/export/{company}")
async def export_results(company: str):
    logger.info(f"Export requested for company: {company}")
    logger.info(f"Available companies in research_results: {list(research_results.keys())}")
    
    if company not in research_results:
        logger.error(f"No research results found for company: {company}")
        raise HTTPException(status_code=404, detail="No research results found for this company")
    
    # Create a formatted export
    export_data = {
        "company": company,
        "research_results": research_results[company]
    }
    
    logger.info(f"Successfully exported JSON data for company: {company}")
    return JSONResponse(content=export_data)

@router.get("/export/{company}/{format}")
async def export_results_format(company: str, format: str):
    logger.info(f"Export requested for company: {company} in format: {format}")
    logger.info(f"Available companies in research_results: {list(research_results.keys())}")

    if company not in research_results:
        raise HTTPException(status_code=404, detail="No research results found for this company")
    
    # Create a formatted export
    export_data = {
        "company": company,
        "research_results": research_results[company]
    }
    
    if format == "json":
        return JSONResponse(content=export_data)
    elif format == "word":
        from docx import Document
        from io import BytesIO
        
        # Create a new Word document
        doc = Document()
        doc.add_heading(f"Research Results for {company}", 0)
        
        # Add each research result to the document
        for result in research_results[company]:
            doc.add_heading(f"Category: {result['promptCategory']} - {result['subcategory']}", 1)
            doc.add_paragraph(f"Prompt: {result['prompt']}")
            doc.add_paragraph(result['result'])
            if result.get('sources'):
                doc.add_heading("Sources:", 2)
                for i, source in enumerate(result['sources']):
                    doc.add_paragraph(f"{i+1}. {source}")
            doc.add_page_break()
        
        # Save the document to a BytesIO object
        f = BytesIO()
        doc.save(f)
        f.seek(0)
        
        # Return the document as a downloadable file
        return StreamingResponse(
            f, 
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename={company}_research.docx"}
        )
    elif format == "pdf":
        from fpdf import FPDF
        from io import BytesIO
        
        # Create a new PDF document
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Set fonts
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"Research Results for {company}", 0, 1, "C")
        
        # Add each research result to the document
        for result in research_results[company]:
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, f"Category: {result['promptCategory']} - {result['subcategory']}", 0, 1)
            
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Prompt:", 0, 1)
            
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, result['prompt'])
            
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Result:", 0, 1)
            
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, result['result'])
            
            if result.get('sources'):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Sources:", 0, 1)
                
                pdf.set_font("Arial", "", 12)
                for i, source in enumerate(result['sources']):
                    pdf.multi_cell(0, 10, f"{i+1}. {source}")
        
        # Save the PDF to a BytesIO object
        f = BytesIO()
        f.write(pdf.output(dest='S').encode('latin1'))
        f.seek(0)
        
        # Return the PDF as a downloadable file
        return StreamingResponse(
            f, 
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={company}_research.pdf"}
        )
    elif format == "excel":
        import pandas as pd
        
        # Create a new Excel workbook
        f = BytesIO()
        writer = pd.ExcelWriter(f, engine='xlsxwriter')
        
        # Create a summary sheet
        summary_data = []
        for result in research_results[company]:
            summary_data.append({
                "Category": result['promptCategory'],
                "Subcategory": result['subcategory'],
                "Prompt": result['prompt'][:100] + "..." if len(result['prompt']) > 100 else result['prompt']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        
        # Create a sheet for each research result
        for i, result in enumerate(research_results[company]):
            sheet_name = f"{result['category']}_{i+1}"
            if len(sheet_name) > 31:  # Excel sheet name length limit
                sheet_name = sheet_name[:31]
            
            # Create a DataFrame for this result
            result_data = {
                "Field": ["Category", "Subcategory", "Prompt", "Result"],
                "Value": [
                    result['category'],
                    result['subcategory'],
                    result['prompt'],
                    result['result']
                ]
            }
            
            # Add sources if available
            if result.get('sources'):
                for i, source in enumerate(result['sources']):
                    result_data["Field"].append(f"Source {i+1}")
                    result_data["Value"].append(source)
            
            result_df = pd.DataFrame(result_data)
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Save the workbook
        writer.close()
        f.seek(0)
        
        # Return the Excel file as a downloadable file
        return StreamingResponse(
            f, 
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={company}_research.xlsx"}
        )
    elif format == "pptx":
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN
        from io import BytesIO
        from collections import defaultdict
        
        # Create a new PowerPoint presentation
        prs = Presentation()
        
        # Define theme colors based on the HTML theme
        THEME_PRIMARY = RGBColor(243, 88, 88)  # #f35858 - accent color
        THEME_GRADIENT_START = RGBColor(255, 112, 136)  # #FF7088 - gradient start
        THEME_GRADIENT_END = RGBColor(242, 182, 157)  # #F2B69D - gradient end
        THEME_BACKGROUND = RGBColor(245, 247, 250)  # #f5f7fa - background color
        THEME_TEXT = RGBColor(82, 87, 92)  # #52575c - text color
        
        # Add title slide
        title_slide_layout = prs.slide_layouts[0]
        slide = prs.slides.add_slide(title_slide_layout)
        
        # Apply theme colors to title slide
        title = slide.shapes.title
        subtitle = slide.placeholders[1]
        
        # Set title text and formatting
        title.text = f"Research Results for {company}"
        title_tf = title.text_frame
        title_tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        title_run = title_tf.paragraphs[0].runs[0]
        title_run.font.color.rgb = THEME_PRIMARY
        title_run.font.bold = True
        title_run.font.size = Pt(36)
        
        # Set subtitle text and formatting
        subtitle.text = f"Generated on {datetime.now().strftime('%Y-%m-%d')}"
        subtitle_tf = subtitle.text_frame
        subtitle_tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        subtitle_run = subtitle_tf.paragraphs[0].runs[0]
        subtitle_run.font.color.rgb = THEME_TEXT
        subtitle_run.font.size = Pt(20)
        
        # Apply background fill to title slide
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = THEME_BACKGROUND
        
        # Organize research results by subcategory
        subcategory_results = defaultdict(list)
        for result in research_results[company]:
            key = f"{result['promptCategory']} - {result['subcategory']}"
            subcategory_results[key].append(result)
        
        # Add content slides - one per subcategory
        content_slide_layout = prs.slide_layouts[1]  # Title and content layout
        
        for subcategory, results in subcategory_results.items():
            slide = prs.slides.add_slide(content_slide_layout)
            
            # Apply background fill to content slide
            background = slide.background
            fill = background.fill
            fill.solid()
            fill.fore_color.rgb = THEME_BACKGROUND
            
            # Set title text and formatting
            title = slide.shapes.title
            title.text = subcategory
            title_tf = title.text_frame
            title_tf.paragraphs[0].alignment = PP_ALIGN.LEFT
            title_run = title_tf.paragraphs[0].runs[0]
            title_run.font.color.rgb = THEME_PRIMARY
            title_run.font.bold = True
            title_run.font.size = Pt(28)
            
            # Add content placeholder
            content = slide.placeholders[1]
            tf = content.text_frame
            
            # Add each prompt and result for this subcategory
            for result in results:
                # Add prompt with styling
                p = tf.add_paragraph()
                p.text = f"Prompt: {result['prompt'][:100]}..." if len(result['prompt']) > 100 else f"Prompt: {result['prompt']}"
                p.alignment = PP_ALIGN.LEFT
                p_run = p.runs[0]
                p_run.font.bold = True
                p_run.font.color.rgb = THEME_GRADIENT_START
                
                # Add result with styling
                p = tf.add_paragraph()
                p.text = result['result'][:500] + "..." if len(result['result']) > 500 else result['result']
                p.alignment = PP_ALIGN.LEFT
                p_run = p.runs[0]
                p_run.font.color.rgb = THEME_TEXT
                
                # Add sources if available
                if result.get('sources') and len(result['sources']) > 0:
                    p = tf.add_paragraph()
                    p.text = "Sources:"
                    p.alignment = PP_ALIGN.LEFT
                    p_run = p.runs[0]
                    p_run.font.bold = True
                    p_run.font.color.rgb = THEME_PRIMARY
                    
                    for source in result['sources'][:3]:  # Limit to first 3 sources to save space
                        p = tf.add_paragraph()
                        p.text = source[:100] + "..." if len(source) > 100 else source
                        p.level = 1  # Indent sources
                        p_run = p.runs[0]
                        p_run.font.italic = True
                        p_run.font.color.rgb = THEME_TEXT
                
                # Add a separator between results (except for the last one)
                if result != results[-1]:
                    p = tf.add_paragraph()
                    p.text = "───────────────────"
                    p.alignment = PP_ALIGN.CENTER
                    p_run = p.runs[0]
                    p_run.font.color.rgb = THEME_GRADIENT_END
        
        # Save the presentation to a BytesIO object
        f = BytesIO()
        prs.save(f)
        f.seek(0)
        
        # Return the PowerPoint file as a downloadable file
        return StreamingResponse(
            f, 
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={"Content-Disposition": f"attachment; filename={company}_research.pptx"}
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")