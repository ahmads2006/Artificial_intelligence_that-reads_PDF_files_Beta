import PyPDF2
import os
import logging
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
from typing import List, Optional, Dict, Any
from datetime import datetime
# إعداد API
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_qa.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_input(text: str, max_length: int = 1000) -> str:
    """Validate and sanitize user input"""
    if not text or not text.strip():
        raise ValueError("Input cannot be empty")

    # Remove potentially harmful characters
    sanitized = ''.join(char for char in text if char.isprintable())
    sanitized = sanitized.strip()

    if len(sanitized) > max_length:
        logger.warning(f"Input too long ({len(sanitized)}), truncating to {max_length}")
        sanitized = sanitized[:max_length]

    return sanitized

def get_ai_response(prompt: str) -> str:
    """Get AI response"""
    try:
        response = genai.generate_content(prompt)
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        return f"Error getting AI response: {e}"


def validate_file_path(file_path: str) -> bool:
    """Validate file path and check if it's a PDF"""
    if not file_path or not os.path.exists(file_path):
        return False

    if not file_path.lower().endswith('.pdf'):
        logger.warning(f"File {file_path} is not a PDF file")
        return False

    # Check file size (max 50MB)
    max_size = 50 * 1024 * 1024  # 50MB
    if os.path.getsize(file_path) > max_size:
        logger.warning(f"File {file_path} is too large ({os.path.getsize(file_path)} bytes)")
        return False

    return True

class AIEnhancedPDFQA:
    def __init__(self, pdf_path=None, api_key=None, use_ai=True):
        """
        Advanced PDF Question Answering System with AI

        Args:
            pdf_path: Path to PDF file (optional)
            api_key: Gemini API key (optional)
            use_ai: Use AI or traditional search
        """
        logger.info("Initializing AIEnhancedPDFQA system")
        self.pdf_text = ""
        self.pdf_path = pdf_path
        self.use_ai = use_ai
        self.ai_model = None
        self.stats = {
            'questions_asked': 0,
            'ai_responses': 0,
            'traditional_responses': 0,
            'last_activity': datetime.now()
        }
        # Rate limiting for API calls (max 10 calls per minute)
        self.api_call_times = []
        self.max_calls_per_minute = 10

        # Cache for summaries to avoid regenerating for same content
        self._summary_cache = {}

        # Initialize AI model
        if use_ai and api_key:
            self._initialize_ai(api_key)

        if pdf_path:
            self.load_pdf(pdf_path)

        logger.info(f"System initialized. AI mode: {self.use_ai}, PDF loaded: {bool(pdf_path)}")
    
    def _initialize_ai(self, api_key):
        """Initialize Gemini model"""
        try:
            logger.info("Initializing Gemini AI model...")
            genai.configure(api_key=api_key)
            self.ai_model = genai.GenerativeModel('gemini-flash-latest')
            logger.info("AI model initialized successfully")
            print("✅ AI mode enabled successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize AI model: {e}")
            print(f"⚠️ Failed to enable AI: {e}")
            print("Using traditional search instead...")
            self.use_ai = False
    
    def load_pdf(self, path):
        """Load PDF file with error handling"""
        try:
            logger.info(f"Attempting to load PDF file: {path}")

            # Validate file path
            if not validate_file_path(path):
                error_msg = f"Invalid PDF file: {path}"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                return False

            if not os.path.exists(path):
                logger.error(f"PDF file not found: {path}")
                raise FileNotFoundError(f"File not found: {path}")

            with open(path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                self.pdf_text = ""
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    self.pdf_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"

            self.pdf_path = path
            # Clear summary cache since we have new content
            self._summary_cache.clear()
            logger.info(f"Successfully loaded PDF with {len(reader.pages)} pages")
            print(f"✔ Loaded {len(reader.pages)} pages from file")
            return True
        except Exception as e:
            logger.error(f"Error loading PDF file {path}: {e}")
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
        """Split text into smaller chunks to avoid model limits with smart boundaries"""
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        paragraphs = text.split('\n\n')  # Split by paragraphs first

        current_chunk = ""
        current_size = 0

        for paragraph in paragraphs:
            paragraph_size = len(paragraph) + 2  # +2 for \n\n

            if current_size + paragraph_size > max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
                current_size = paragraph_size
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
                current_size += paragraph_size

        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # If we still have very large chunks, split by sentences
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chunk_size:
                # Split by sentences
                sentences = chunk.split('. ')
                temp_chunk = ""
                temp_size = 0

                for sentence in sentences:
                    sentence_size = len(sentence) + 2  # +2 for '. '

                    if temp_size + sentence_size > max_chunk_size and temp_chunk:
                        final_chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                        temp_size = sentence_size
                    else:
                        if temp_chunk:
                            temp_chunk += '. ' + sentence
                        else:
                            temp_chunk = sentence
                        temp_size += sentence_size

                if temp_chunk:
                    final_chunks.append(temp_chunk.strip())
            else:
                final_chunks.append(chunk)

        logger.debug(f"Text chunked into {len(final_chunks)} chunks")
        return final_chunks
    
    def answer_with_ai(self, question: str) -> str:
        """Answer question using AI"""
        logger.info(f"Processing AI question: {question[:100]}...")

        if not self.ai_model:
            logger.warning("AI model not available for question")
            return "❌ AI model not available. Make sure you entered a valid API key."

        if not self.pdf_text:
            logger.warning("No PDF text available for AI question")
            return "⚠ No PDF file loaded yet."

        # Check rate limit
        if not self._check_rate_limit():
            logger.warning("Rate limit exceeded for AI question")
            return "❌ Rate limit exceeded. Please wait a moment before asking another question."
        
        try:
            # Update statistics
            self.stats['questions_asked'] += 1
            self.stats['last_activity'] = datetime.now()

            # Split text if it's too large
            chunks = self._chunk_text(self.pdf_text)
            logger.debug(f"Text split into {len(chunks)} chunks")

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
                self.stats['ai_responses'] += 1
                logger.info("AI response generated successfully")
                return response.text

            # If text is large, search in each chunk
            else:
                print(f"🔍 Searching in {len(chunks)} sections of the document...")
                best_answer = None

                for i, chunk in enumerate(chunks):
                    print(f"  📖 Searching section {i+1}/{len(chunks)}...", end="", flush=True)
                    prompt = f"""Question: {question}

Part of the document:
{chunk}

Do you find an answer to the question in this part? If yes, answer accurately. If no, say "No answer in this part"."""

                    response = self.ai_model.generate_content(prompt)
                    answer = response.text

                    if "No answer" not in answer and "no answer" not in answer.lower() and len(answer) > 50:
                        best_answer = answer
                        logger.info(f"Found answer in chunk {i+1}")
                        print(" ✅ Found!")
                        break
                    else:
                        print(" ❌ No answer", end="")

                print()  # New line after progress

                if best_answer:
                    self.stats['ai_responses'] += 1
                    logger.info("AI chunk search successful")
                else:
                    logger.warning("No clear answer found in any chunk")

                return best_answer if best_answer else "❌ Could not find a clear answer in the document."

        except Exception as e:
            logger.error(f"AI error during question processing: {e}")
            print(f"❌ AI Error: {e}")
            return "An error occurred while processing the question. Try rephrasing."
    
    def answer_traditional(self, question: str, context_lines: int = 3) -> str:
        """Traditional answer by searching for keywords"""
        logger.info(f"Processing traditional question: {question[:100]}...")

        if not self.pdf_text:
            logger.warning("No PDF text available for traditional question")
            return "⚠ No PDF file loaded yet."
        
        sentences = [s.strip() for s in self.pdf_text.replace("\n", " ").split(".") if s.strip()]
        question_words = [w.lower() for w in question.split() if len(w) > 2]
        matches = []

        for i, sentence in enumerate(sentences):
            match_count = sum(1 for word in question_words if word in sentence.lower())
            if match_count >= 1:
                matches.append((i, match_count, sentence))

        if not matches:
            logger.info(f"No matches found for traditional search with keywords: {question_words}")
            return "❌ Could not find an answer. Try using different keywords."

        matches.sort(key=lambda x: x[1], reverse=True)
        best_idx, match_count, _ = matches[0]

        self.stats['traditional_responses'] += 1
        self.stats['questions_asked'] += 1
        self.stats['last_activity'] = datetime.now()

        logger.info(f"Traditional search found {len(matches)} matches, best match has {match_count} keyword matches")

        start = max(0, best_idx - context_lines)
        end = min(len(sentences), best_idx + context_lines + 1)
        context = ". ".join(sentences[start:end])

        return context + "."
    
    def answer_question(self, question: str) -> str:
        """Answer question (AI or traditional)"""
        try:
            # Validate and sanitize input
            clean_question = validate_input(question)
            logger.info(f"Processing question: {clean_question[:50]}...")

            if self.use_ai and self.ai_model:
                return self.answer_with_ai(clean_question)
            else:
                return self.answer_traditional(clean_question)
        except ValueError as e:
            logger.warning(f"Invalid question input: {e}")
            return f"❌ Invalid input: {e}"
        except Exception as e:
            logger.error(f"Unexpected error in answer_question: {e}")
            return "❌ An unexpected error occurred. Please try again."
    
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
        """Get intelligent summary using AI with caching"""
        if not self.ai_model:
            return self.get_traditional_summary()

        if not self.pdf_text:
            return "No text to summarize."

        # Create cache key based on PDF content hash and max_length
        import hashlib
        content_hash = hashlib.md5(self.pdf_text[:50000].encode('utf-8')).hexdigest()
        cache_key = f"{content_hash}_{max_length}"

        # Check cache first
        if cache_key in self._summary_cache:
            logger.info("Using cached AI summary")
            return self._summary_cache[cache_key]

        try:
            logger.info("Generating AI summary...")
            # Use part of the text for summary
            text_sample = self.pdf_text[:50000]  # First 50k characters

            prompt = f"""Summarize the following document comprehensively and concisely in approximately {max_length} words:

{text_sample}

Summary:"""

            response = self.ai_model.generate_content(prompt)
            summary = response.text

            # Cache the result
            self._summary_cache[cache_key] = summary
            logger.info("AI summary generated and cached successfully")
            return summary
        except Exception as e:
            logger.error(f"AI summary error: {e}")
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
            logger.info(f"Switched to mode: {mode}")
            print(f"🔄 Switched to mode: {mode}")
        else:
            logger.warning("Attempted to toggle AI mode but AI not available")
            print("⚠️ AI not available. Make sure you entered a valid API key.")

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            **self.stats,
            'pdf_loaded': bool(self.pdf_text),
            'ai_available': bool(self.ai_model),
            'pdf_path': self.pdf_path,
            'text_length': len(self.pdf_text) if self.pdf_text else 0
        }

    def _check_rate_limit(self) -> bool:
        """Check if API call is within rate limits"""
        now = datetime.now()
        # Remove calls older than 1 minute
        self.api_call_times = [t for t in self.api_call_times if (now - t).total_seconds() < 60]

        if len(self.api_call_times) >= self.max_calls_per_minute:
            logger.warning("Rate limit exceeded")
            return False

        self.api_call_times.append(now)
        return True


def interactive_mode():
    """Enhanced interactive mode with AI support"""
    print("\n" + "=" * 80)
    print("🤖 Smart PDF Question Answering System - Powered by Gemini AI")
    print("📚 Ask questions about your PDF documents with AI assistance")
    print("=" * 80)

    # Automatic API setup
    api_key = os.getenv("GEMINI_API_KEY") or "########################"
    print("\n🔑 Initializing system...")

    qa = AIEnhancedPDFQA(api_key=api_key, use_ai=True)

    if qa.ai_model:
        print("✅ AI mode enabled successfully!")
    else:
        print("⚠️ AI mode not available - using traditional search")

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
    print("\n" + "=" * 80)
    print("📋 Available Commands:")
    print("  💬  Ask any question directly - get smart AI-powered answers")
    print("  🔍  search:keyword - search for specific keywords in the document")
    print("  📄  summary - generate intelligent document summary")
    print("  📊  stats - show system statistics and usage info")
    print("  🔄  toggle - switch between AI and traditional search modes")
    print("  📁  load - load a different PDF file")
    print("  ❓  help - show this help message")
    print("  🚪  exit - quit the application")
    print("=" * 80)

    while True:
        user_input = input("\n💬 Enter command or question: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == "exit":
            stats = qa.get_stats()
            print("\n📊 Session Summary:")
            print(f"  • Total questions: {stats['questions_asked']}")
            print(f"  • AI responses: {stats['ai_responses']}")
            print(f"  • Traditional responses: {stats['traditional_responses']}")
            print("👋 Thank you for using the Smart PDF QA System!")
            break

        elif user_input.lower() == "help":
            print("\n" + "=" * 60)
            print("📖 Help - How to use the system:")
            print("=" * 60)
            print("💬 Questions:")
            print("  • Type any question in natural language")
            print("  • Examples: 'What are the main concepts?', 'Explain UML diagrams'")
            print()
            print("🔍 Search Commands:")
            print("  • search:keyword - Find all occurrences of a word")
            print("  • Example: 'search:class' or 'search:UML'")
            print()
            print("📄 Document Analysis:")
            print("  • summary - Generate AI-powered document summary")
            print("  • stats - Show usage statistics and system info")
            print()
            print("⚙️ System Controls:")
            print("  • toggle - Switch between AI and traditional modes")
            print("  • load - Load a different PDF file")
            print("  • exit - Quit the application")
            print("=" * 60)

        elif user_input.lower() == "summary":
            print("\n📝 Generating summary...")
            if qa.use_ai and qa.ai_model:
                summary = qa.get_ai_summary()
            else:
                summary = qa.get_traditional_summary()
            print(f"\n📄 Summary:\n{summary}")

        elif user_input.lower() == "toggle":
            qa.toggle_ai_mode()

        elif user_input.lower() == "stats":
            stats = qa.get_stats()
            print("\n" + "=" * 50)
            print("📊 System Statistics")
            print("=" * 50)
            print(f"📝 Questions Asked: {stats['questions_asked']}")
            print(f"🤖 AI Responses: {stats['ai_responses']}")
            print(f"🔍 Traditional Responses: {stats['traditional_responses']}")
            print(f"📄 PDF File: {'✅ Loaded' if stats['pdf_loaded'] else '❌ Not loaded'}")
            if stats['pdf_loaded'] and stats['pdf_path']:
                print(f"   📁 Path: {stats['pdf_path']}")
            print(f"🧠 AI Model: {'✅ Available' if stats['ai_available'] else '❌ Not available'}")
            print(f"📊 Text Length: {stats['text_length']:,} characters")
            print(f"🕒 Last Activity: {stats['last_activity'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⚡ Current Mode: {'AI Mode' if qa.use_ai else 'Traditional Mode'}")
            print("=" * 50)

        elif user_input.lower().startswith("search:"):
            keyword = user_input[7:].strip()
            try:
                clean_keyword = validate_input(keyword, max_length=100)
                results = qa.search_keyword(clean_keyword)
                if results:
                    print(f"\n🔍 Found {len(results)} results:")
                    for result in results[:15]:
                        print(f"  • {result}")
                else:
                    print("❌ No results found.")
            except ValueError as e:
                print(f"❌ Invalid keyword: {e}")

        elif user_input.lower() == "load":
            path = input("Enter new PDF file path: ").strip()
            try:
                clean_path = validate_input(path, max_length=500)
                qa.load_pdf(clean_path)
            except ValueError as e:
                print(f"❌ Invalid file path: {e}")

        else:
            print("\n🤔 Searching for answer...")
            answer = qa.answer_question(user_input)
            print(f"\n✨ Answer:\n{answer}")


if __name__ == "__main__":
    interactive_mode()