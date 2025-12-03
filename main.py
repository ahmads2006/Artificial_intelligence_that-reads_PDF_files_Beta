import PyPDF2
import os
from pathlib import Path
import google.generativeai as genai
from typing import List, Optional

class AIEnhancedPDFQA:
    def __init__(self, pdf_path=None, api_key=None, use_ai=True):
        """
        Advanced PDF Question Answering System with AI
        
        Args:
            pdf_path: Path to PDF file (optional)
            api_key: Gemini API key (optional)
            use_ai: Use AI or traditional search
        """
        self.pdf_text = ""
        self.pdf_path = pdf_path
        self.use_ai = use_ai
        self.ai_model = None
        
        # Initialize AI model
        if use_ai and api_key:
            self._initialize_ai(api_key)
        
        if pdf_path:
            self.load_pdf(pdf_path)
    
    def _initialize_ai(self, api_key):
        """Initialize Gemini model"""
        try:
            genai.configure(api_key=api_key)
            self.ai_model = genai.GenerativeModel('gemini-flash-latest')
            print("✅ AI mode enabled successfully!")
        except Exception as e:
            print(f"⚠️ Failed to enable AI: {e}")
            print("Using traditional search instead...")
            self.use_ai = False
    
    def load_pdf(self, path):
        """Load PDF file with error handling"""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            with open(path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                self.pdf_text = ""
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    self.pdf_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"

            self.pdf_path = path
            print(f"✔ Loaded {len(reader.pages)} pages from file")
            return True
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return False
    
    def load_multiple_pdfs(self, paths: List[str]):
        """Load multiple PDF files"""
        all_text = ""
        total_pages = 0
        
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as file:
                        reader = PyPDF2.PdfReader(file)
                        all_text += f"\n\n===== File: {os.path.basename(path)} =====\n\n"
                        for page_num, page in enumerate(reader.pages):
                            all_text += f"\n--- Page {page_num + 1} ---\n{page.extract_text()}\n"
                        total_pages += len(reader.pages)
                    print(f"✔ Loaded: {path} ({len(reader.pages)} pages)")
                except Exception as e:
                    print(f"❌ Error in {path}: {e}")
        
        self.pdf_text = all_text
        print(f"✅ Total pages loaded: {total_pages}")
        return bool(all_text)
    
    def _chunk_text(self, text: str, max_chunk_size: int = 30000) -> List[str]:
        """Split text into smaller chunks to avoid model limits"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word) + 1
            if current_size > max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def answer_with_ai(self, question: str) -> str:
        """Answer question using AI"""
        if not self.ai_model:
            return "❌ AI model not available. Make sure you entered a valid API key."

        if not self.pdf_text:
            return "⚠ No PDF file loaded yet."
        
        try:
            # Split text if it's too large
            chunks = self._chunk_text(self.pdf_text)
            
            # If there's only one chunk
            if len(chunks) == 1:
                prompt = f"""You are an intelligent assistant specialized in answering questions from documents.

Document:
{self.pdf_text}

Question: {question}

Instructions:
1. Read the document carefully
2. Answer the question accurately based only on the document content
3. If you don't find the answer in the document, say so clearly
4. Make your answer clear and direct

Answer:"""
                
                response = self.ai_model.generate_content(prompt)
                return response.text
            
            # If text is large, search in each chunk
            else:
                print(f"🔍 Searching in {len(chunks)} sections of the document...")
                best_answer = None

                for i, chunk in enumerate(chunks):
                    prompt = f"""Question: {question}

Part of the document:
{chunk}

Do you find an answer to the question in this part? If yes, answer accurately. If no, say "No answer in this part"."""
                    
                    response = self.ai_model.generate_content(prompt)
                    answer = response.text
                    
                    if "No answer" not in answer and "no answer" not in answer.lower() and len(answer) > 50:
                        best_answer = answer
                        break
                
                return best_answer if best_answer else "❌ Could not find a clear answer in the document."

        except Exception as e:
            print(f"❌ AI Error: {e}")
            return "An error occurred while processing the question. Try rephrasing."
    
    def answer_traditional(self, question: str, context_lines: int = 3) -> str:
        """Traditional answer by searching for keywords"""
        if not self.pdf_text:
            return "⚠ No PDF file loaded yet."
        
        sentences = [s.strip() for s in self.pdf_text.replace("\n", " ").split(".") if s.strip()]
        question_words = [w.lower() for w in question.split() if len(w) > 2]
        matches = []
        
        for i, sentence in enumerate(sentences):
            match_count = sum(1 for word in question_words if word in sentence.lower())
            if match_count >= 1:
                matches.append((i, match_count, sentence))
        
        if not matches:
            return "❌ Could not find an answer. Try using different keywords."
        
        matches.sort(key=lambda x: x[1], reverse=True)
        best_idx, _, _ = matches[0]
        
        start = max(0, best_idx - context_lines)
        end = min(len(sentences), best_idx + context_lines + 1)
        context = ". ".join(sentences[start:end])
        
        return context + "."
    
    def answer_question(self, question: str) -> str:
        """Answer question (AI or traditional)"""
        if self.use_ai and self.ai_model:
            return self.answer_with_ai(question)
        else:
            return self.answer_traditional(question)
    
    def search_keyword(self, keyword: str) -> List[str]:
        """Search for a keyword in the text"""
        if not self.pdf_text:
            return []
        
        results = []
        lines = self.pdf_text.split("\n")
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                results.append(f"[Line {i}]: {line.strip()}")
        
        return results
    
    def get_ai_summary(self, max_length: int = 300) -> str:
        """Get intelligent summary using AI"""
        if not self.ai_model:
            return self.get_traditional_summary()

        if not self.pdf_text:
            return "No text to summarize."
        
        try:
            # Use part of the text for summary
            text_sample = self.pdf_text[:50000]  # First 50k characters
            
            prompt = f"""Summarize the following document comprehensively and concisely in approximately {max_length} words:

{text_sample}

Summary:"""
            
            response = self.ai_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"❌ AI Summary Error: {e}")
            return self.get_traditional_summary()

    def get_traditional_summary(self, max_sentences: int = 5) -> str:
        """Traditional summary"""
        if not self.pdf_text:
            return "No text to summarize."
        
        sentences = [s.strip() for s in self.pdf_text.split(".") if len(s.strip()) > 50]
        return ". ".join(sentences[:max_sentences]) + "."
    
    def toggle_ai_mode(self):
        """Toggle AI mode"""
        if self.ai_model:
            self.use_ai = not self.use_ai
            mode = "Smart AI" if self.use_ai else "Traditional"
            print(f"🔄 Switched to mode: {mode}")
        else:
            print("⚠️ AI not available. Make sure you entered a valid API key.")


def interactive_mode():
    """Enhanced interactive mode with AI support"""
    print("=" * 70)
    print("🤖 Smart PDF Question Answering System - Powered by Gemini AI")
    print("=" * 70)

    # Automatic API setup
    api_key = "GEMINI_API_KEY"
    print("🔑 AI mode enabled automatically!")

    qa = AIEnhancedPDFQA(api_key=api_key, use_ai=True)

    # Load default PDF automatically
    default_pdf = "data/UML.pdf"
    if os.path.exists(default_pdf):
        print(f"\n📂 Loading default file: {default_pdf}")
        if not qa.load_pdf(default_pdf):
            print("❌ Failed to load default file. You can load another file.")
    else:
        print("\n📂 Default PDF file not found.")
        print("You can load a PDF file using the 'load' command")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("📋 Available commands:")
    print("  - Type your question directly to get a smart answer")
    print("  - search:keyword - to search for a specific keyword")
    print("  - summary - to get an intelligent summary")
    print("  - toggle - to switch between AI and traditional search")
    print("  - load - to load a new file")
    print("  - help - show help")
    print("  - exit - to quit")
    print("=" * 70)

    while True:
        user_input = input("\n💬 Enter command or question: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == "exit":
            print("👋 Thanks for using the system!")
            break
        
        elif user_input.lower() == "help":
            print("\n📖 Help:")
            print("  • Ask any question about the file content")
            print("  • Use 'search:keyword' for quick search")
            print("  • Use 'summary' to summarize the document")
            print("  • Use 'toggle' to switch between AI and regular search")

        elif user_input.lower() == "summary":
            print("\n📝 Generating summary...")
            if qa.use_ai and qa.ai_model:
                summary = qa.get_ai_summary()
            else:
                summary = qa.get_traditional_summary()
            print(f"\n📄 Summary:\n{summary}")

        elif user_input.lower() == "toggle":
            qa.toggle_ai_mode()

        elif user_input.lower().startswith("search:"):
            keyword = user_input[7:].strip()
            results = qa.search_keyword(keyword)
            if results:
                print(f"\n🔍 Found {len(results)} results:")
                for result in results[:15]:
                    print(f"  • {result}")
            else:
                print("❌ No results found.")

        elif user_input.lower() == "load":
            path = input("Enter new PDF file path: ").strip()
            qa.load_pdf(path)

        else:
            print("\n🤔 Searching for answer...")
            answer = qa.answer_question(user_input)
            print(f"\n✨ Answer:\n{answer}")


if __name__ == "__main__":
    interactive_mode()