import subprocess
import logging

def scan_ports(domain):
    try:
        result = subprocess.run(['nmap', '-sT', domain], check=True, text=True, capture_output=True)
        logging.info(f"Port scan results for {domain}: {result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Port scanning failed for {domain}. Error: {e}")
        return ""
