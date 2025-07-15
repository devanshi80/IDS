import socket
import psutil

def scan_ports():
    open_ports = []
    for conn in psutil.net_connections(kind='inet'):
        if conn.status == 'LISTEN' and conn.laddr:
            port = conn.laddr.port
            open_ports.append(port)
    return open_ports

def build_feature_vector(ports):
    return [{"port": p} for p in ports]
