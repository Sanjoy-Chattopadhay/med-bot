import os
from dotenv import load_dotenv


def check_setup():
    print("🔍 Checking Med Bot Setup...")
    print("=" * 40)

    # Check .env file
    if os.path.exists('.env'):
        print("✅ .env file found")
        load_dotenv()

        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            if groq_key == "your_actual_groq_api_key_here":
                print("❌ GROQ API key is still placeholder")
                print("   Please replace with your actual API key")
            else:
                print("✅ GROQ API key configured")
        else:
            print("❌ GROQ API key not found in .env")
    else:
        print("❌ .env file not found")
        print("   Please create a .env file with your GROQ API key")

    # Check data directory
    if os.path.exists('data'):
        pdf_files = [f for f in os.listdir('data') if f.endswith('.pdf')]
        if pdf_files:
            print(f"✅ Found {len(pdf_files)} PDF files in data directory")
            for pdf in pdf_files:
                print(f"   - {pdf}")
        else:
            print("❌ No PDF files found in data directory")
    else:
        print("❌ data directory not found")
        print("   Please create a 'data' folder and add your medical PDF files")

    # Check vector store
    if os.path.exists('vectorstore/db_faiss'):
        print("✅ Vector store found")
    else:
        print("❌ Vector store not found")
        print("   Please run create_memory_for_LLM.py first")

    print("=" * 40)
    print("Setup instructions:")
    print("1. Get API key from: https://console.groq.com/")
    print("2. Add API key to .env file")
    print("3. Add PDF files to data/ directory")
    print("4. Run create_memory_for_LLM.py")
    print("5. Run streamlit run medical-chatbot.py")


if __name__ == "__main__":
    check_setup()