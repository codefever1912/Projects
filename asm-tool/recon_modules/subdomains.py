import subprocess
import logging

def get_subdomains(domain):
    try:
        result = subprocess.run(['subfinder', '-d', domain, '-o', 'subdomains.txt'], check=True, text=True, capture_output=True)
        logging.info(f"Subdomains found for {domain}: {result.stdout}")
        with open("subdomains.txt", "r") as file:
            return file.readlines()
    except subprocess.CalledProcessError as e:
        logging.error(f"Subdomain enumeration failed for {domain}. Error: {e}")
        return []
