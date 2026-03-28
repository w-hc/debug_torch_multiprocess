DEBUGPY_ENABLE=0 torchrun --nproc_per_node=2 train.py

# archimedes firewall blocks direct TCP connection to port 5678. error "No route to host"
# forward compute-node port 5678 to vscode node 
ssh -N -L 5678:localhost:5678 whc@archimedes.ttic.edu &

# test port 5678 is reachable
python -c "import socket; s=socket.socket(); s.settimeout(3); s.connect(('localhost', 5678)); print('OK'); s.close()"

DEBUGPY_ENABLE=1 torchrun --nproc_per_node=2 train.py