# Aegis Distributed Computing Framework - Lab Prototype

## Setup Instructions

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start first node (creates network):
   ```bash
   python main.py --mode create --port 5000 --name Hospital_A --verbose
   ```

4. Note the Network ID displayed and start additional nodes:
   ```bash
   python main.py --mode join --port 5001 --network-id <NETWORK_ID> --name Hospital_B --verbose
   python main.py --mode join --port 5002 --network-id <NETWORK_ID> --name Hospital_C --verbose
   ```

## Interactive Commands

Once in the interactive shell:
- `status` - Show node status
- `peers` - List connected peers
- `consensus` - Initiate consensus round
- `train` - Start federated training
- `audit` - Display blockchain audit trail
- `privacy` - Show privacy budget status
- `help` - Show all commands
- `exit` - Shutdown node

## Troubleshooting

- Port already in use: Change --port argument
- Cannot connect to peers: Check firewall settings
- Consensus fails: Ensure at least 3 nodes connected (f < n/3)
