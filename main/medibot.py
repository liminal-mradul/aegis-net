#!/usr/bin/env python3
"""
Full-Featured MediBot System - Complete Aegis Integration
Includes ALL Aegis features + Medical AI capabilities
FIXED VERSION with improved error handling and stability
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq
import argparse
import os
import sys
import json
import time
import base64
from pathlib import Path
import traceback
import threading
import PyPDF2
import io
import numpy as np
import socket

sys.path.insert(0, str(Path(__file__).parent))

from src.node import AegisNode
from src.config import NodeConfig
from src.utils.logger import setup_logger

app = Flask(__name__, static_folder='static')
CORS(app)

# Global state
groq_client = None
aegis_node = None
shared_records_db = []
public_health_data = []
logger = None

MEDICAL_KB = {
    "diabetes": """Diabetes: chronic condition affecting blood sugar processing. Type 1: no insulin production. Type 2: insulin resistance. Symptoms: thirst, frequent urination, fatigue, blurred vision. Management: monitoring, diet, exercise, medications.""",
    "hypertension": """High blood pressure: consistently elevated BP (>=130/80). Often no symptoms. Risk factors: obesity, salt, lack of exercise, stress. Management: lifestyle changes, ACE inhibitors, beta blockers.""",
    "heart_disease": """Cardiovascular disease: affects heart and blood vessels. Types: coronary artery disease, heart failure, arrhythmias. Risk factors: high cholesterol, hypertension, smoking, diabetes. Prevention: healthy diet, exercise.""",
}

def print_banner():
    """Print ASCII banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                                   ║
║     ██████╗ ███████╗ ██████╗ ██╗███████╗                          ║
║    ██╔══██╗██╔════╝██╔════╝ ██║██╔════╝                          ║
║    ███████║█████╗  ██║  ███╗██║███████╗                          ║
║    ██╔══██║██╔══╝  ██║   ██║██║╚════██║                          ║
║    ██║  ██║███████╗╚██████╔╝██║███████║                          ║
║    ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝╚══════╝                          ║
║                                                                   ║
║              MediBot - Medical Intelligence System                ║
║         Privacy-Preserving Federated Healthcare Network          ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def get_public_ip():
    """Get public IP and local IP"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        
        try:
            import urllib.request
            public_ip = urllib.request.urlopen('https://api.ipify.org', timeout=3).read().decode('utf8')
        except:
            public_ip = "Unable to fetch"
        
        return local_ip, public_ip
    except:
        return "127.0.0.1", "Unable to fetch"

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return None

def anonymize_record(record):
    """Apply differential privacy to medical record"""
    anonymized = record.copy()
    
    # Remove identifiers
    anonymized.pop('patient_id', None)
    anonymized.pop('patient_name', None)
    
    # Age grouping
    if 'age' in anonymized:
        age = anonymized['age']
        if age < 18:
            anonymized['age_group'] = '<18'
        elif age < 30:
            anonymized['age_group'] = '18-29'
        elif age < 50:
            anonymized['age_group'] = '30-49'
        elif age < 65:
            anonymized['age_group'] = '50-64'
        else:
            anonymized['age_group'] = '65+'
        del anonymized['age']
    
    # Add Laplace noise to numerical values
    for key, value in list(anonymized.items()):  # Use list() to avoid dict size change during iteration
        if isinstance(value, (int, float)) and key not in ['age_group']:
            try:
                epsilon = 0.1
                sensitivity = abs(value) * 0.1 if value != 0 else 1.0
                scale = sensitivity / epsilon
                noise = np.random.laplace(0, scale)
                anonymized[key] = float(value + noise)
            except Exception as e:
                logger.warning(f"Failed to add noise to {key}: {e}")
                pass
    
    return anonymized

def log_to_aegis(action_type, data_summary):
    """Log action to blockchain"""
    if not aegis_node:
        return
    
    try:
        aegis_node.blockchain.add_transaction({
            'type': f'medibot_{action_type}',
            'timestamp': time.time(),
            'data_hash': hash(str(data_summary)),
            'privacy_preserved': True
        }, auto_mine=False)
    except Exception as e:
        logger.warning(f"Aegis log error: {e}")

def groq_generate(prompt, model='llama-3.3-70b-versatile', max_tokens=1024, temperature=0.7):
    """Generate using Groq API with error handling"""
    if not groq_client:
        return None, "Groq API not configured. Use --groq-key to enable."
    
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a medical education assistant. Provide clear, well-formatted information using markdown. Always remind users to consult healthcare professionals for medical decisions."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content, None
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return None, str(e)

def rag_retrieve(query):
    """RAG retrieval from medical knowledge base"""
    query_lower = query.lower()
    relevant_docs = []
    
    for topic, content in MEDICAL_KB.items():
        if topic in query_lower:
            relevant_docs.append(content)
    
    if not relevant_docs:
        return "General medical information: Consult healthcare professionals for specific advice."
    
    return "\n\n".join(relevant_docs[:2])

def analyze_public_health_trends():
    """Analyze trends across all records"""
    if not public_health_data:
        return None
    
    analysis = {
        'total_records': len(public_health_data),
        'conditions': {},
        'age_groups': {}
    }
    
    for record in public_health_data:
        condition = record.get('condition', 'unknown')
        analysis['conditions'][condition] = analysis['conditions'].get(condition, 0) + 1
        
        age_group = record.get('age_group', 'unknown')
        analysis['age_groups'][age_group] = analysis['age_groups'].get(age_group, 0) + 1
    
    analysis['conditions'] = dict(sorted(analysis['conditions'].items(), key=lambda x: x[1], reverse=True))
    
    return analysis

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve main page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/system/info')
@app.route('/api/status')
def system_info():
    """Get complete system information"""
    try:
        local_ip, public_ip = get_public_ip()
        
        info = {
            'status': 'running',
            'local_ip': local_ip,
            'public_ip': public_ip,
            'web_url': f'http://{local_ip}:{app.config.get("WEB_PORT", 8080)}',
            'public_url': f'http://{public_ip}:{app.config.get("WEB_PORT", 8080)}' if public_ip != "Unable to fetch" else None,
            'groq_available': groq_client is not None,
            'aegis_available': aegis_node is not None,
            'shared_records': len(shared_records_db),
            'public_health_records': len(public_health_data),
            'timestamp': time.time()
        }
        
        # Add Aegis network info
        if aegis_node:
            try:
                network_info = aegis_node.get_network_info()
                blockchain_summary = aegis_node.blockchain.get_chain_summary()
                privacy_budget = aegis_node.budget_manager.get_remaining_budget()
                
                # Safely get network ID
                network_id_value = 'N/A'
                if hasattr(aegis_node, 'network_config') and aegis_node.network_config:
                    network_id_value = aegis_node.network_config.network_id
                elif 'network_id' in network_info:
                    network_id_value = network_info['network_id']
                
                info['aegis_info'] = {
                    'available': True,
                    'node_id': network_info['node_id'][:16] + '...',
                    'node_name': network_info['node_name'],
                    'network_id': network_id_value,
                    'peers': network_info['connected_peers'],
                    'blockchain_length': blockchain_summary['length'],
                    'blockchain_transactions': blockchain_summary['total_transactions'],
                    'privacy_epsilon_remaining': round(privacy_budget['epsilon_remaining'], 2),
                    'privacy_utilization': round(privacy_budget['epsilon_utilization'] * 100, 1)
                }
            except Exception as e:
                logger.error(f"Error getting Aegis info: {e}")
                info['aegis_info'] = {'available': True, 'peers': 0, 'error': str(e)}
        else:
            info['aegis_info'] = {'available': False, 'peers': 0}
        
        return jsonify(info)
    except Exception as e:
        logger.error(f"System info error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize', methods=['POST'])
def summarize_report():
    """Summarize medical report"""
    try:
        report_text = None
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.pdf'):
                report_text = extract_text_from_pdf(io.BytesIO(file.read()))
                if not report_text:
                    return jsonify({'error': 'Could not extract text from PDF'}), 400
        else:
            data = request.json
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            report_text = data.get('text', '')
        
        if not report_text:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(report_text) > 10000:
            report_text = report_text[:10000] + "..."
        
        prompt = f"""Summarize this medical report in patient-friendly language using clear markdown formatting:

**Report:**
{report_text}

Please provide:
1. **Key Findings** - Main diagnoses or observations
2. **Important Values** - Critical measurements or test results
3. **Recommendations** - Suggested follow-up or treatments
4. **Note** - Reminder to discuss with healthcare provider

Use markdown formatting with headers, lists, and emphasis where appropriate."""
        
        summary, error = groq_generate(prompt, max_tokens=800, temperature=0.5)
        
        if error:
            return jsonify({'error': error}), 500
        
        log_to_aegis('report_summarization', {'length': len(report_text)})
        
        return jsonify({
            'summary': summary,
            'model': 'llama-3.3-70b-versatile'
        })
    
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Health chatbot with markdown support"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message'}), 400
        
        is_trend_query = any(word in user_message.lower() for word in 
            ['trend', 'common', 'frequent', 'pattern', 'statistics', 'data', 'regional'])
        
        if is_trend_query and public_health_data:
            trends = analyze_public_health_trends()
            context = f"""**Public Health Data:**
- Total Records: {trends['total_records']}
- **Top Conditions:** {', '.join(list(trends['conditions'].keys())[:3])}
- **Age Distribution:** {len(trends['age_groups'])} groups

**Detailed Breakdown:**
```json
{json.dumps(trends, indent=2)}
```"""
            
            prompt = f"""Analyze this anonymized health data and provide insights in well-formatted markdown:

{context}

**Question:** {user_message}

Please provide:
1. **Summary** - Key observations from the data
2. **Trends** - Notable patterns
3. **Insights** - What this might indicate
4. **Privacy Note** - Remind that data is privacy-preserved

Use markdown formatting with headers, lists, code blocks, and emphasis."""
        else:
            context = rag_retrieve(user_message)
            prompt = f"""**Medical Knowledge Base:**
{context}

**Question:** {user_message}

Please provide a comprehensive answer using markdown formatting:
- Use **bold** for important terms
- Use bullet points for lists
- Use headers (##) for sections
- Use `code blocks` for medical values/measurements
- Always remind users to consult healthcare professionals

Answer:"""
        
        response, error = groq_generate(prompt, max_tokens=600, temperature=0.7)
        
        if error:
            return jsonify({'error': error}), 500
        
        log_to_aegis('chatbot_query', {'length': len(user_message)})
        
        return jsonify({
            'response': response,
            'model': 'llama-3.3-70b-versatile',
            'data_enhanced': is_trend_query
        })
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/records/share', methods=['POST'])
def share_medical_record():
    """Share anonymized record"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        record = data.get('record', {})
        share_publicly = data.get('public', False)
        
        if not record:
            return jsonify({'error': 'No record data'}), 400
        
        anonymized = anonymize_record(record)
        anonymized['timestamp'] = time.time()
        anonymized['hospital_id'] = aegis_node.node_id[:8] if aegis_node else 'unknown'
        
        shared_records_db.append(anonymized)
        
        if share_publicly:
            public_health_data.append(anonymized)
        
        log_to_aegis('record_shared', {'record_id': anonymized.get('record_id', 'unknown'), 'public': share_publicly})
        
        if aegis_node:
            try:
                from src.network.protocol import Message
                message = Message(
                    msg_type='medical_record_shared',
                    sender_id=aegis_node.node_id,
                    data={'record': anonymized, 'public': share_publicly},
                    timestamp=time.time()
                )
                aegis_node.peer_manager.broadcast_message(message)
            except Exception as e:
                logger.warning(f"Failed to broadcast record: {e}")
        
        return jsonify({
            'success': True,
            'record_id': anonymized.get('record_id'),
            'anonymized': True,
            'public': share_publicly
        })
    
    except Exception as e:
        logger.error(f"Record sharing error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/records/search', methods=['POST'])
def search_records():
    """Search shared records"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        query = data.get('query', {})
        
        results = []
        for record in shared_records_db:
            match = True
            for key, value in query.items():
                if key in record:
                    if isinstance(value, str):
                        if value.lower() not in str(record[key]).lower():
                            match = False
                            break
                    else:
                        if record[key] != value:
                            match = False
                            break
            
            if match:
                results.append(record)
        
        return jsonify({'success': True, 'count': len(results), 'records': results[:50]})
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health/trends')
def public_health_trends():
    """Get public health trends"""
    try:
        if not public_health_data:
            return jsonify({'message': 'No public health data yet', 'total_records': 0})
        
        trends = analyze_public_health_trends()
        
        return jsonify({
            'success': True,
            'trends': trends,
            'data_sources': len(set(r.get('hospital_id') for r in public_health_data if 'hospital_id' in r))
        })
    
    except Exception as e:
        logger.error(f"Trends error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/aegis/peers')
def aegis_peers():
    """Get connected peers"""
    if not aegis_node:
        return jsonify({'error': 'Aegis not available'}), 400
    
    try:
        active = aegis_node.peer_manager.get_active_peers()
        peers = []
        
        for peer_id in active:
            peer = aegis_node.peer_manager.peers.get(peer_id, {})
            peers.append({
                'id': peer_id[:16] + '...',
                'url': peer.get('url'),
                'last_seen': time.time() - peer.get('last_seen', 0)
            })
        
        return jsonify({'total': len(peers), 'peers': peers})
    except Exception as e:
        logger.error(f"Peers error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/aegis/connect', methods=['POST'])
def aegis_connect_peer():
    """Connect to peer"""
    if not aegis_node:
        return jsonify({'error': 'Aegis not available'}), 400
    
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        peer_url = data.get('url')
        
        if not peer_url:
            return jsonify({'error': 'No URL'}), 400
        
        success = aegis_node.connect_to_peer(peer_url)
        
        return jsonify({'success': success, 'message': 'Connected' if success else 'Failed'})
    except Exception as e:
        logger.error(f"Connect error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/aegis/mine', methods=['POST'])
def aegis_mine():
    """Mine blockchain"""
    if not aegis_node:
        return jsonify({'error': 'Aegis not available'}), 400
    
    try:
        aegis_node.blockchain.force_mine()
        summary = aegis_node.blockchain.get_chain_summary()
        
        return jsonify({
            'success': True,
            'length': summary['length'],
            'transactions': summary['total_transactions']
        })
    except Exception as e:
        logger.error(f"Mining error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/aegis/train', methods=['POST'])
def aegis_train():
    """Start federated training"""
    if not aegis_node:
        return jsonify({'error': 'Aegis not available'}), 400
    
    try:
        data = request.json
        if not data:
            rounds = 3
        else:
            rounds = data.get('rounds', 3)
        
        if rounds < 1 or rounds > 10:
            return jsonify({'error': 'Rounds must be 1-10'}), 400
        
        def train():
            try:
                aegis_node.run_federated_training(rounds)
            except Exception as e:
                logger.error(f"Training error: {e}")
        
        thread = threading.Thread(target=train, daemon=True)
        thread.start()
        
        return jsonify({'success': True, 'message': f'Started {rounds} rounds'})
    except Exception as e:
        logger.error(f"Training start error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/aegis/blockchain')
def aegis_blockchain():
    """Get blockchain transactions"""
    if not aegis_node:
        return jsonify({'error': 'Aegis not available'}), 400
    
    try:
        history = aegis_node.blockchain.get_transaction_history()
        return jsonify({'total': len(history), 'transactions': history[-20:]})
    except Exception as e:
        logger.error(f"Blockchain error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/aegis/model/evaluate')
def aegis_model_evaluate():
    """Evaluate model"""
    if not aegis_node:
        return jsonify({'error': 'Aegis not available'}), 400
    
    try:
        X, y = aegis_node.data_loader.get_full_dataset()
        metrics = aegis_node.model.evaluate(X, y)
        
        return jsonify({'success': True, 'metrics': metrics})
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# INITIALIZATION
# ============================================================================

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='MediBot - Full-Featured Medical AI System with Aegis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Create new network with web interface on port 8080
  python3 medibot.py --mode create --port 5000 --web-port 8080 --name Hospital_A
  
  # Join existing network with custom ports
  python3 medibot.py --mode join --port 5001 --web-port 8081 --name Hospital_B --network <ID>
  
  # With CSV data and Groq API
  python3 medibot.py --mode create --port 5000 --web-port 8080 --name Hospital_A \\
                     --data-file hospital_a.csv --groq-key <YOUR_KEY>
        '''
    )
    
    # Required
    parser.add_argument('--mode', type=str, required=True, choices=['create', 'join'],
                       help='Mode: create new network or join existing')
    parser.add_argument('--port', type=int, required=True,
                       help='Aegis P2P port (e.g., 5000)')
    parser.add_argument('--name', type=str, required=True,
                       help='Node name (e.g., Hospital_A)')
    
    # Optional
    parser.add_argument('--web-port', type=int, default=None,
                       help='Web interface port (default: aegis_port + 100)')
    parser.add_argument('--network', type=str,
                       help='Network ID to join (required for join mode)')
    parser.add_argument('--data-file', type=str,
                       help='Path to CSV data file')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address (default: 0.0.0.0)')
    parser.add_argument('--groq-key', type=str,
                       help='Groq API key for AI features')
    parser.add_argument('--stake', type=float, default=100.0,
                       help='Initial stake (default: 100.0)')
    parser.add_argument('--reputation', type=float, default=0.8,
                       help='Initial reputation (default: 0.8)')
    parser.add_argument('--epsilon', type=float, default=10.0,
                       help='Privacy budget epsilon (default: 10.0)')
    parser.add_argument('--delta', type=float, default=1e-5,
                       help='Privacy parameter delta (default: 1e-5)')
    parser.add_argument('--data-size', type=int, default=500,
                       help='Synthetic data size (default: 500)')
    parser.add_argument('--num-features', type=int, default=20,
                       help='Number of features (default: 20)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Validation
    if args.mode == 'join' and not args.network:
        parser.error("--network is required when using --mode join")
    
    if args.data_file and not Path(args.data_file).exists():
        parser.error(f"Data file not found: {args.data_file}")
    
    if not (1024 <= args.port <= 65535):
        parser.error(f"Port must be between 1024-65535, got {args.port}")
    
    # Set default web port
    if args.web_port is None:
        args.web_port = args.port + 100
    
    return args

def init_system(args):
    """Initialize the complete system"""
    global aegis_node, groq_client, logger
    
    try:
        # Setup logger
        logger = setup_logger('medibot', 'DEBUG' if args.verbose else 'INFO')
        
        # Initialize Groq if key provided
        if args.groq_key:
            try:
                groq_client = Groq(api_key=args.groq_key)
                logger.info("✓ Groq API initialized")
            except Exception as e:
                logger.warning(f"⚠ Groq init failed: {e}")
        else:
            logger.info("⚠ Groq API not configured (use --groq-key to enable)")
        
        # Create Aegis node configuration
        config = NodeConfig(
            port=args.port,
            host=args.host,
            node_name=args.name,
            stake=args.stake,
            reputation=args.reputation,
            epsilon_total=args.epsilon,
            delta_total=args.delta,
            data_file=args.data_file,
            data_size=args.data_size,
            num_features=args.num_features,
            verbose=args.verbose
        )
        
        # Initialize Aegis node
        aegis_node = AegisNode(config)
        logger.info("✓ Aegis node initialized")
        
        # Create or join network
        if args.mode == 'create':
            network_id = aegis_node.create_network()
            logger.info("="*70)
            logger.info(f"✓ NETWORK CREATED")
            logger.info(f"  Network ID: {network_id}")
            logger.info("="*70)
        else:
            aegis_node.join_network(args.network)
            logger.info(f"✓ Joined network: {args.network}")
        
        # Get network info
        local_ip, public_ip = get_public_ip()
        
        # Get network ID safely
        network_id = 'N/A'
        if aegis_node and hasattr(aegis_node, 'network_config') and aegis_node.network_config:
            network_id = aegis_node.network_config.network_id
        
        # Print summary
        print("\n" + "="*70)
        print("SYSTEM READY")
        print("="*70)
        print(f"Node Name:    {args.name}")
        print(f"Node ID:      {aegis_node.node_id[:16]}...")
        print(f"Network ID:   {network_id}")
        print(f"\nLocal IP:     {local_ip}")
        print(f"Public IP:    {public_ip}")
        print(f"\nAegis P2P:    http://{local_ip}:{args.port}")
        print(f"Web UI:       http://{local_ip}:{args.web_port}")
        if public_ip != "Unable to fetch":
            print(f"Public Web:   http://{public_ip}:{args.web_port}")
        print("\nFeatures:")
        print(f"  ✓ Blockchain Audit Trail")
        print(f"  ✓ Federated Learning")
        print(f"  ✓ Differential Privacy")
        print(f"  ✓ Secure Aggregation")
        print(f"  {'✓' if groq_client else '✗'} Medical AI (Groq)")
        print("="*70 + "\n")
        
        # Store ports in app config
        app.config['AEGIS_PORT'] = args.port
        app.config['WEB_PORT'] = args.web_port
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Initialization failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    print_banner()
    
    # Parse arguments
    args = parse_arguments()
    
    # Initialize system
    if not init_system(args):
        print("✗ Failed to initialize system")
        sys.exit(1)
    
    # Start Flask web server
    print(f"Starting web server on port {args.web_port}...")
    print(f"Open browser: http://localhost:{args.web_port}")
    print("="*70 + "\n")
    
    # Suppress Flask startup messages
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.logger.setLevel(logging.ERROR)
    
    try:
        app.run(
            host=args.host,
            port=args.web_port,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n✓ Shutting down gracefully...")
    except Exception as e:
        print(f"✗ Server error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
