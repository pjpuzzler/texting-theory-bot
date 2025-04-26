import base64
import os
from cryptography.fernet import Fernet
import argparse

def generate_key():
    key = Fernet.generate_key()
    return key.decode()

def encrypt_prompt(prompt_text, key):
    key_bytes = key.encode()
    
    f = Fernet(key_bytes)
    
    encrypted_prompt = f.encrypt(prompt_text.encode())
    
    return base64.b64encode(encrypted_prompt).decode()

def decrypt_prompt(encrypted_text, key):
    key_bytes = key.encode()
    
    f = Fernet(key_bytes)
    
    encrypted_bytes = base64.b64decode(encrypted_text.encode())
    decrypted_prompt = f.decrypt(encrypted_bytes).decode()
    
    return decrypted_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encrypt or decrypt a prompt using a key")
    parser.add_argument("action", choices=["generate", "encrypt", "decrypt"], 
                        help="Action to perform: generate key, encrypt prompt, or decrypt prompt")
    parser.add_argument("--prompt_file", help="File containing the prompt to encrypt/decrypt")
    parser.add_argument("--output_file", help="File to write the result to")
    parser.add_argument("--key", help="Encryption key (or will use KEY env var if not provided)")
    
    args = parser.parse_args()
    
    if args.action == "generate":
        key = generate_key()
        print(f"Generated key: {key}")
        print("Store this key securely as a GitHub secret!")
        
    elif args.action in ["encrypt", "decrypt"]:
        key = args.key or os.environ.get("PROMPT_KEY")
        if not key:
            print("Error: No key provided. Use --key or set PROMPT_KEY environment variable.")
            exit(1)
            
        if not args.prompt_file:
            print(f"Error: No prompt file provided for {args.action}ion.")
            exit(1)
            
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if args.action == "encrypt":
            result = encrypt_prompt(text, key)
        else:
            result = decrypt_prompt(text, key)
            
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"{args.action.capitalize()}ed content written to {args.output_file}")
        else:
            print(f"{args.action.capitalize()}ed content:")
            print(result)