# asm_tool/main.py

from input_parser import parse_input
from recon_modules.subdomains import SubdomainEnumerator

def main():
    domains = parse_input("example.com")  # You can change to input.csv if testing
    sub_enum = SubdomainEnumerator()

    for domain in domains:
        print(f"\n🔍 Enumerating subdomains for: {domain}")
        subdomains = sub_enum.enumerate(domain)
        print(f"✅ Found {len(subdomains)} subdomains")
        for sub in subdomains:
            print(f"  - {sub}")

if __name__ == "__main__":
    main()
