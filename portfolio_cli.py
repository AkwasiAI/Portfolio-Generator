#!/usr/bin/env python3
"""
Portfolio Management CLI Tool
A command-line tool to upload and retrieve financial documents to/from Firestore.

Usage:
  portfolio_cli.py upload <filename> --type=<type> [--format=<format>] [--not-latest]
  portfolio_cli.py get-latest --type=<type> [--output=<output_file>]
  portfolio_cli.py list [--limit=<num>] [--type=<type>]
  portfolio_cli.py (-h | --help)

Options:
  -h --help           Show this help message
  --type=<type>       Document type (reports, portfolio_weights, report_feedback)
  --format=<format>   File format (markdown, json) [default: auto]
  --output=<file>     Output file to save the content
  --limit=<num>       Limit number of results [default: 10]
  --not-latest        Mark the uploaded file as not the latest version
"""

import os
import sys
import json
import datetime
from pathlib import Path
from docopt import docopt
from google.cloud import firestore
import tabulate
from rich.console import Console
from rich.syntax import Syntax

from src.portfolio_generator.firestore_uploader import FirestoreUploader

console = Console()

class PortfolioManager:
    def __init__(self, database='hedgefundintelligence'):
        # Initialize Firestore client
        try:
            self.uploader = FirestoreUploader(database=database)
        except Exception as e:
            console.print(f"[bold red]Error connecting to Firestore: {str(e)}[/bold red]")
            console.print("[yellow]Make sure you have set GOOGLE_APPLICATION_CREDENTIALS environment variable[/yellow]")
            sys.exit(1)

    def upload_file(self, filename, doc_type, file_format='auto', is_latest=True):
        """Upload a file to Firestore and mark it as the latest version"""
        if not os.path.exists(filename):
            console.print(f"[bold red]Error: File {filename} not found[/bold red]")
            return False

        success = self.uploader.upload_file(filename, doc_type, file_format, is_latest)
        if success:
            console.print(f"[bold green]Successfully uploaded {filename} to Firestore[/bold green]")
        return success

    def get_latest(self, doc_type, output_file=None):
        """Retrieve the latest document of the specified type"""
        try:
            query = self.uploader.collection.where('doc_type', '==', doc_type).where('is_latest', '==', True).limit(1)
            docs = list(query.stream())
            
            if not docs:
                console.print(f"[bold yellow]No {doc_type} document found[/bold yellow]")
                return False
            
            doc_data = docs[0].to_dict()
            content = doc_data['content']
            file_format = doc_data.get('file_format', 'text')
            
            # Format output based on file format
            if file_format == 'json' and isinstance(content, dict):
                formatted_content = json.dumps(content, indent=2)
                syntax = Syntax(formatted_content, "json", theme="monokai", line_numbers=True)
            elif file_format == 'markdown':
                formatted_content = str(content)
                syntax = Syntax(formatted_content, "markdown", theme="monokai", line_numbers=True)
            else:
                formatted_content = str(content)
                syntax = Syntax(formatted_content, "text", theme="monokai", line_numbers=True)
            
            # Save to file if specified
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    if file_format == 'json' and isinstance(content, dict):
                        json.dump(content, f, indent=2)
                    else:
                        f.write(str(content))
                console.print(f"[bold green]Saved latest {doc_type} to {output_file}[/bold green]")
            else:
                # Print to console
                console.print(f"[bold blue]Latest {doc_type} document:[/bold blue]")
                console.print(syntax)
                
            return True
            
        except Exception as e:
            console.print(f"[bold red]Error retrieving latest portfolio: {str(e)}[/bold red]")
            return False

    def list_portfolios(self, limit=10, doc_type=None):
        """List all documents with their metadata, optionally filtered by document type"""
        try:
            if doc_type:
                query = self.uploader.collection.where('doc_type', '==', doc_type).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
            else:
                query = self.uploader.collection.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
            docs = list(query.stream())
            
            if not docs:
                console.print("[bold yellow]No portfolios found[/bold yellow]")
                return False
            
            table_data = []
            for doc in docs:
                data = doc.to_dict()
                timestamp = data.get('timestamp')
                if timestamp:
                    timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    
                table_data.append([
                    doc.id[:8] + '...',
                    data.get('filename', 'N/A'),
                    data.get('doc_type', 'N/A'),
                    data.get('file_format', 'N/A'),
                    timestamp,
                    '✓' if data.get('is_latest') else ''
                ])
            
            headers = ["ID", "Filename", "Document Type", "Format", "Timestamp", "Latest"]
            console.print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))
            
            return True
            
        except Exception as e:
            console.print(f"[bold red]Error listing portfolios: {str(e)}[/bold red]")
            return False

def main():
    """Main entry point for the CLI"""
    arguments = docopt(__doc__)
    manager = PortfolioManager()
    
    try:
        if arguments['upload']:
            filename = arguments['<filename>']
            file_type = arguments['--type']
            
            valid_types = ['reports', 'portfolio_weights', 'report_feedback']
            if file_type not in valid_types:
                console.print(f"[bold red]Error: Document type must be one of {', '.join(valid_types)}[/bold red]")
                return 1
                
            is_latest = not arguments['--not-latest']
            manager.upload_file(filename, file_type, arguments['--format'], is_latest)
            
        elif arguments['get-latest']:
            doc_type = arguments['--type']
            output_file = arguments['--output']
            
            valid_types = ['reports', 'portfolio_weights', 'report_feedback']
            if doc_type not in valid_types:
                console.print(f"[bold red]Error: Document type must be one of {', '.join(valid_types)}[/bold red]")
                return 1
                
            manager.get_latest(doc_type, output_file)
            
        elif arguments['list']:
            limit = int(arguments['--limit'])
            doc_type = arguments['--type']
            manager.list_portfolios(limit, doc_type)
            
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        return 1
        
    return 0

if __name__ == '__main__':
    sys.exit(main())
