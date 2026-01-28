#!/usr/bin/env python3
"""
Convert Markdown Documentation to PDF
"""

import markdown
import pdfkit
from pathlib import Path

def convert_markdown_to_pdf():
    """Convert the project documentation to PDF format"""
    
    # Input and output paths
    md_file = Path("project_documentation.md")
    pdf_file = Path("Micro_Expression_Recognition_Documentation.pdf")
    
    if not md_file.exists():
        print(f"❌ Markdown file not found: {md_file}")
        return False
    
    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['tables', 'toc', 'fenced_code'])
    
    # Add CSS styling for better PDF output
    html_with_css = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Micro-Expression Recognition System Documentation</title>
        <style>
            body {{
                font-family: 'Times New Roman', serif;
                line-height: 1.6;
                margin: 2cm;
                color: #333;
            }}
            h1 {{
                page-break-before: always;
                text-align: center;
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 5px;
                margin-top: 30px;
            }}
            h3 {{
                color: #7f8c8d;
                margin-top: 25px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            code {{
                background-color: #f8f9fa;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                border-left: 4px solid #3498db;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                margin-left: 0;
                padding-left: 20px;
                font-style: italic;
                color: #7f8c8d;
            }}
            .toc {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .toc ul {{
                list-style-type: none;
                padding-left: 0;
            }}
            .toc li {{
                margin: 5px 0;
            }}
            .toc a {{
                text-decoration: none;
                color: #3498db;
            }}
            .toc a:hover {{
                text-decoration: underline;
            }}
            @page {{
                size: A4;
                margin: 2cm;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # PDF generation options
    options = {
        'page-size': 'A4',
        'margin-top': '2cm',
        'margin-right': '2cm',
        'margin-bottom': '2cm',
        'margin-left': '2cm',
        'encoding': "UTF-8",
        'no-outline': None,
        'enable-local-file-access': None,
        'javascript-delay': 2000,
        'load-error-handling': 'ignore',
        'load-media-error-handling': 'ignore'
    }
    
    try:
        # Convert to PDF
        pdfkit.from_string(html_with_css, str(pdf_file), options=options)
        print(f"✅ PDF generated successfully: {pdf_file}")
        print(f"📄 File size: {pdf_file.stat().st_size / 1024:.1f} KB")
        return True
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        print("💡 Make sure wkhtmltopdf is installed:")
        print("   - Windows: Download from https://wkhtmltopdf.org/")
        print("   - Mac: brew install wkhtmltopdf")
        print("   - Linux: sudo apt-get install wkhtmltopdf")
        return False

if __name__ == "__main__":
    print("🔄 Converting documentation to PDF...")
    success = convert_markdown_to_pdf()
    
    if success:
        print("\n🎉 PDF conversion completed!")
        print("📁 Output file: Micro_Expression_Recognition_Documentation.pdf")
    else:
        print("\n❌ PDF conversion failed. Please check the error messages above.")
