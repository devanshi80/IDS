def map_port_to_feature(port):
    # Simplified port mapping (could be enriched later)
    common_ports = {
        22: "SSH", 80: "HTTP", 443: "HTTPS", 53: "DNS", 25: "SMTP"
    }
    return {"port": port, "service": common_ports.get(port, "OTHER")}
