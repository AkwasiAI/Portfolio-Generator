#!/usr/bin/env python3
"""
Firestore Uploader Module for Portfolio Generator
This module handles uploading portfolio reports and weights to Firestore.
"""

import os
import json
import datetime
import glob
from pathlib import Path
from google.cloud import firestore

class FirestoreUploader:
    def __init__(self, database='hedgefundintelligence'):
        """Initialize Firestore client with the specified database"""
        try:
            # First, check if GOOGLE_APPLICATION_CREDENTIALS is already set
            credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            
            # If not set, attempt to find credentials file in project directory
            if not credentials_path or not os.path.exists(credentials_path):
                print("GOOGLE_APPLICATION_CREDENTIALS not set or file not found, searching for credential files...")
                # Look for credential files in the project directory
                project_root = Path(__file__).parent.parent.parent  # Go up to project root
                
                # Check for JSON files that might be service account keys
                credential_files = []
                
                # Check for *-firebase-adminsdk-*.json pattern (Firebase pattern)
                credential_files.extend(glob.glob(str(project_root / "*-firebase-adminsdk-*.json")))
                
                # Check for files named service-account*.json
                credential_files.extend(glob.glob(str(project_root / "service-account*.json")))
                
                # Check for files containing 'credentials' in the name
                credential_files.extend(glob.glob(str(project_root / "*credential*.json")))
                
                # Check for hedgefundintelligence specific files
                credential_files.extend(glob.glob(str(project_root / "hedgefundintelligence*.json")))
                
                # Check for any JSON file that might contain 'private_key' (common in service account files)
                for json_file in glob.glob(str(project_root / "*.json")):
                    try:
                        with open(json_file, 'r') as f:
                            content = f.read()
                            if 'private_key' in content or 'type":"service_account"' in content:
                                credential_files.append(json_file)
                    except:
                        pass  # Skip files we can't read
                
                if credential_files:
                    # Take the first matching file
                    credentials_path = credential_files[0]
                    print(f"Found potential credentials file: {credentials_path}")
                    # Set the environment variable
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
                else:
                    print("No credential files found in the project directory")
            
            # Initialize Firestore client
            self.db = firestore.Client(database=database)
            self.collection = self.db.collection('portfolios')
            print(f"Successfully connected to Firestore database: {database}")
        except Exception as e:
            print(f"Error connecting to Firestore: {str(e)}")
            print("Make sure you have set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            print("or place your service account JSON file in the project root directory")
            raise

    def upload_file(self, filename, doc_type, file_format='auto', is_latest=True):
        """Upload a file to Firestore and mark it as the latest version"""
        if not os.path.exists(filename):
            print(f"Error: File {filename} not found")
            return False

        try:
            # Determine file format if set to auto
            if file_format == 'auto':
                extension = Path(filename).suffix.lower()
                if extension == '.json':
                    file_format = 'json'
                elif extension in ['.md', '.markdown']:
                    file_format = 'markdown'
                else:
                    print(f"Warning: Could not determine file format from extension. Treating as text.")
                    file_format = 'text'
            
            # Read file content
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Process content based on format
            if file_format == 'json':
                try:
                    content_json = json.loads(content)
                    content = content_json  # Store as parsed JSON in Firestore
                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON format - {str(e)}")
                    return False
            
            # Add document with timestamp and content
            doc_ref = self.collection.document()
            doc_ref.set({
                'content': content,
                'doc_type': doc_type,  # 'reports', 'portfolio_weights', 'report_feedback'
                'file_format': file_format,  # 'markdown', 'json', 'text'
                'filename': os.path.basename(filename),
                'timestamp': firestore.SERVER_TIMESTAMP,
                'is_latest': is_latest
            })
            
            # Mark previous documents as not latest if this one is latest
            if is_latest:
                self._update_latest_flags(doc_ref.id, doc_type)
            
            print(f"Successfully uploaded {filename} to Firestore")
            print(f"Document ID: {doc_ref.id}")
            return True
            
        except Exception as e:
            print(f"Error uploading file: {str(e)}")
            return False

    def _update_latest_flags(self, current_doc_id, doc_type):
        """Set is_latest=False for all documents of the same type except the current one"""
        query = self.collection.where('doc_type', '==', doc_type).where('is_latest', '==', True)
        results = query.stream()
        
        batch = self.db.batch()
        for doc in results:
            if doc.id != current_doc_id:
                doc_ref = self.collection.document(doc.id)
                batch.update(doc_ref, {'is_latest': False})
        
        batch.commit()
        
    def upload_portfolio_data(self, report_path, portfolio_data_path):
        """
        Upload both the portfolio report (markdown) and portfolio weights (JSON) to Firestore
        
        Args:
            report_path: Path to the markdown report file
            portfolio_data_path: Path to the portfolio JSON data file
        
        Returns:
            tuple: (report_success, weights_success) indicating if each upload succeeded
        """
        report_success = self.upload_file(report_path, 'reports', file_format='markdown', is_latest=True)
        weights_success = self.upload_file(portfolio_data_path, 'portfolio_weights', file_format='json', is_latest=True)
        
        return report_success, weights_success
