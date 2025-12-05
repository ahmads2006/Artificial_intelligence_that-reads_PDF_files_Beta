import PyPDF2
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import re

load_dotenv()
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
        
        logger.info(f"System initialized - AI: {self.use_ai}, Provider: {self.ai_provider.provider_name}")
    
    def load_pdf(self, path: str, doc_id: Optional[str] = None) -> bool:
        """Load PDF with enhanced error handling and metadata"""
        try:
            # Validate file
            is_valid, message = validate_file_path(path)
            if not is_valid:
                logger.error(f"Validation failed: {message}")
                print(f"❌ {message}")
                return False
            
            # Generate doc_id if not provided
            if doc_id is None:
                doc_id = Path(path).stem
            
            logger.info(f"Loading PDF: {path}")
            start_time = datetime.now()
            
            with open(path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                
                # Extract text with page markers
                text_parts = []
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    text_parts.append(f"\n--- Page {i+1} ---\n{page_text}\n")
                
                full_text = "".join(text_parts)
            
            # Store document and metadata
            self.documents[doc_id] = full_text
            self.metadata[doc_id] = DocumentMetadata(
                filename=Path(path).name,
                num_pages=num_pages,
                text_length=len(full_text),
                load_time=datetime.now(),
                file_size=Path(path).stat().st_size,
                hash=calculate_text_hash(full_text)
            )
            
            self.current_doc_id = doc_id
            self.stats['total_docs_loaded'] += 1
            
            # Clear caches for this document
            self.qa_cache.clear()
            self.summary_cache.clear()
            
            load_duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"PDF loaded successfully in {load_duration:.2f}s - {num_pages} pages")
            print(f"✅ Loaded '{Path(path).name}' - {num_pages} pages ({len(full_text):,} chars)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading PDF {path}: {e}")
            print(f"❌ Error loading PDF: {e}")
            return False
    
    def load_multiple_pdfs(self, paths: List[str]) -> int:
        """Load multiple PDFs"""
        success_count = 0
        
        for path in paths:
            if self.load_pdf(path):
                success_count += 1
        
        print(f"\n📚 Loaded {success_count}/{len(paths)} documents successfully")
        return success_count
    
    def _get_current_text(self) -> str:
        """Get current document text"""
        if not self.current_doc_id or self.current_doc_id not in self.documents:
            return ""
        return self.documents[self.current_doc_id]
    
    def _chunk_text(self, text: str, max_chunk_size: int = 40000) -> List[str]:
        """Smart text chunking with paragraph and sentence boundaries"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para) + 2
            
            if current_size + para_size > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
                current_size = para_size
            else:
                current_chunk = (current_chunk + '\n\n' + para) if current_chunk else para
                current_size += para_size
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Handle oversized chunks
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                temp_chunk = ""
                
                for sent in sentences:
                    if len(temp_chunk) + len(sent) > max_chunk_size and temp_chunk:
                        final_chunks.append(temp_chunk.strip())
                        temp_chunk = sent
                    else:
                        temp_chunk = (temp_chunk + ' ' + sent) if temp_chunk else sent
                
                if temp_chunk:
                    final_chunks.append(temp_chunk.strip())
            else:
                final_chunks.append(chunk)
        
        logger.debug(f"Text chunked into {len(final_chunks)} chunks")
        return final_chunks
    
    def _check_rate_limit(self) -> bool:
        """Check API rate limit with exponential backoff"""
        now = datetime.now()
        self.api_call_times = [t for t in self.api_call_times if (now - t).total_seconds() < 60]
        
        if len(self.api_call_times) >= self.max_calls_per_minute:
            logger.warning("Rate limit reached")
            return False
        
        self.api_call_times.append(now)
        return True
    
    def _calculate_confidence(self, text: str, question: str) -> float:
        """Calculate answer confidence score"""
        question_keywords = set(extract_keywords(question))
        text_keywords = set(extract_keywords(text))
        
        if not question_keywords:
            return 0.0
        
        # Keyword overlap
        overlap = len(question_keywords & text_keywords)
        keyword_score = overlap / len(question_keywords)
        
        # Length penalty (too short answers are suspicious)
        length_score = min(len(text) / 200, 1.0)
        
        # Combined score
        confidence = (keyword_score * 0.7 + length_score * 0.3)
        
        return round(confidence, 2)
    
    def answer_with_ai(self, question: str) -> Tuple[str, float]:
        """Answer question using AI with confidence scoring"""
        text = self._get_current_text()
        
        if not text:
            return "⚠️ No document loaded", 0.0
        
        if not self.ai_provider.available:
            return "⚠️ AI not available. Falling back to traditional search...", 0.0
        
        if not self._check_rate_limit():
            return "⚠️ Rate limit reached. Please wait a moment.", 0.0
        
        try:
            chunks = self._chunk_text(text)
            
            if len(chunks) == 1:
                # Single chunk - direct answer
                prompt = f"""You are an expert document analyst. Answer the question based ONLY on the document provided.

Document:
{text}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the document
- If the answer is not in the document, state: "Information not found in document"
- Be concise but complete
- Cite specific details when possible

Answer:"""
                
                response = self.ai_provider.generate(prompt)
                
                if response:
                    confidence = self._calculate_confidence(response, question)
                    self.stats['ai_responses'] += 1
                    return response, confidence
            
            else:
                # Multi-chunk search
                print(f"🔍 Searching {len(chunks)} document sections...")
                best_answer = None
                best_confidence = 0.0
                
                for i, chunk in enumerate(chunks):
                    prompt = f"""Question: {question}

Document Section:
{chunk}

Does this section contain information to answer the question? If yes, provide the answer. If no, respond with "NO_ANSWER".

Answer:"""
                    
                    response = self.ai_provider.generate(prompt)
                    
                    if response and "NO_ANSWER" not in response:
                        confidence = self._calculate_confidence(response, question)
                        
                        if confidence > best_confidence:
                            best_answer = response
                            best_confidence = confidence
                        
                        if confidence > 0.7:  # Good enough answer found
                            break
                
                if best_answer:
                    self.stats['ai_responses'] += 1
                    return best_answer, best_confidence
            
            return "❌ Could not find relevant information in the document.", 0.0
            
        except Exception as e:
            logger.error(f"AI error: {e}")
            return f"❌ AI Error: {str(e)}", 0.0
    
    def answer_traditional(self, question: str, context_lines: int = 2) -> Tuple[str, float]:
        """Enhanced traditional keyword search with confidence scoring"""
        text = self._get_current_text()
        
        if not text:
            return "⚠️ No document loaded", 0.0
        
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ')) if len(s.strip()) > 20]
        
        # Extract question keywords
        question_keywords = extract_keywords(question)
        
        if not question_keywords:
            return "❌ Please provide a more specific question", 0.0
        
        # Score each sentence
        matches = []
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            match_count = sum(1 for kw in question_keywords if kw in sent_lower)
            
            if match_count > 0:
                # Calculate page number
                page_num = text[:text.find(sent)].count('--- Page') + 1
                matches.append((i, match_count, sent, page_num))
        
        if not matches:
            return "❌ No relevant information found. Try different keywords.", 0.0
        
        # Sort by match count
        matches.sort(key=lambda x: x[1], reverse=True)
        best_idx, match_count, _, page_num = matches[0]
        
        # Get context
        start = max(0, best_idx - context_lines)
        end = min(len(sentences), best_idx + context_lines + 1)
        context = " ".join(sentences[start:end])
        
        # Calculate confidence
        confidence = min(match_count / len(question_keywords), 1.0)
        
        self.stats['traditional_responses'] += 1
        
        result = f"{context}\n\n📍 Source: Page {page_num} (approx.)"
        return result, confidence
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer question with automatic mode selection and caching
        
        Returns structured response with metadata
        """
        try:
            # Update stats
            self.stats['questions_asked'] += 1
            self.stats['last_activity'] = datetime.now()
            
            # Validate input
            clean_question = validate_input(question)
            
            # Check cache
            cache_key = f"{self.current_doc_id}:{calculate_text_hash(clean_question)}"
            cached = self.qa_cache.get(cache_key)
            
            if cached:
                self.stats['cached_responses'] += 1
                logger.info("Returning cached response")
                return {
                    **cached,
                    'cached': True,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Get answer
            if self.use_ai and self.ai_provider.available:
                answer, confidence = self.answer_with_ai(clean_question)
                mode = "AI"
                
                # Fallback to traditional if AI fails
                if confidence < 0.3 or "not found" in answer.lower() or "error" in answer.lower():
                    logger.info("AI confidence low, trying traditional search")
                    answer_trad, conf_trad = self.answer_traditional(clean_question)
                    
                    if conf_trad > confidence:
                        answer = answer_trad
                        confidence = conf_trad
                        mode = "Traditional (AI Fallback)"
            else:
                answer, confidence = self.answer_traditional(clean_question)
                mode = "Traditional"
            
            # Prepare response
            response = {
                'question': clean_question,
                'answer': answer,
                'confidence': confidence,
                'mode': mode,
                'document': self.metadata[self.current_doc_id].filename if self.current_doc_id else None,
                'cached': False,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache response
            self.qa_cache.set(cache_key, response)
            
            return response
            
        except ValueError as e:
            logger.warning(f"Invalid input: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {
                'error': f"Unexpected error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def get_summary(self, max_length: int = 400) -> str:
        """Generate intelligent summary with caching"""
        text = self._get_current_text()
        
        if not text:
            return "No document loaded."
        
        # Check cache
        cache_key = f"{self.current_doc_id}:summary:{max_length}"
        cached = self.summary_cache.get(cache_key)
        
        if cached:
            logger.info("Returning cached summary")
            return cached
        
        # Generate summary
        if self.use_ai and self.ai_provider.available:
            try:
                sample = text[:60000]  # Use more text for better summary
                
                prompt = f"""Provide a comprehensive summary of this document in approximately {max_length} words.

Document:
{sample}

Focus on:
1. Main topics and themes
2. Key concepts and definitions  
3. Important conclusions or findings

Summary:"""
                
                summary = self.ai_provider.generate(prompt)
                
                if summary:
                    self.summary_cache.set(cache_key, summary)
                    return summary
            except Exception as e:
                logger.error(f"AI summary error: {e}")
        
        # Fallback to traditional summary
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 50]
        summary = '. '.join(sentences[:7]) + '.'
        self.summary_cache.set(cache_key, summary)
        
        return summary
    
    def search_keyword(self, keyword: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Advanced keyword search with context and page numbers"""
        text = self._get_current_text()
        
        if not text:
            return []
        
        results = []
        lines = text.split('\n')
        current_page = 1
        
        for i, line in enumerate(lines):
            # Track page numbers
            if '--- Page' in line:
                try:
                    current_page = int(re.search(r'--- Page (\d+)', line).group(1))
                except:
                    pass
                continue
            
            if keyword.lower() in line.lower():
                # Get context
                context_start = max(0, i - 1)
                context_end = min(len(lines), i + 2)
                context = ' '.join(lines[context_start:context_end])
                
                results.append({
                    'line_number': i,
                    'page': current_page,
                    'content': line.strip(),
                    'context': context.strip()
                })
                
                if len(results) >= max_results:
                    break
        
        return results
    
    def toggle_ai_mode(self) -> str:
        """Toggle between AI and traditional modes"""
        if not self.ai_provider.available:
            return "⚠️ AI not available"
        
        self.use_ai = not self.use_ai
        mode = "AI Mode ✨" if self.use_ai else "Traditional Mode 🔍"
        logger.info(f"Switched to: {mode}")
        
        return f"🔄 Switched to: {mode}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Comprehensive system statistics"""
        uptime = datetime.now() - self.stats['start_time']
        
        return {
            'session': {
                'questions_asked': self.stats['questions_asked'],
                'ai_responses': self.stats['ai_responses'],
                'traditional_responses': self.stats['traditional_responses'],
                'cached_responses': self.stats['cached_responses'],
                'uptime': str(uptime).split('.')[0],
                'last_activity': self.stats['last_activity'].strftime('%Y-%m-%d %H:%M:%S')
            },
            'documents': {
                'total_loaded': self.stats['total_docs_loaded'],
                'currently_loaded': len(self.documents),
                'current_doc': self.metadata[self.current_doc_id].filename if self.current_doc_id else None
            },
            'ai': {
                'available': self.ai_provider.available,
                'provider': self.ai_provider.provider_name,
                'mode': 'AI' if self.use_ai else 'Traditional'
            },
            'cache': {
                'qa_cache': self.qa_cache.get_stats(),
                'summary_cache': self.summary_cache.get_stats()
            }
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all loaded documents with metadata"""
        return [
            {
                'id': doc_id,
                'filename': meta.filename,
                'pages': meta.num_pages,
                'size': f"{meta.file_size / 1024:.1f} KB",
                'loaded': meta.load_time.strftime('%Y-%m-%d %H:%M'),
                'current': doc_id == self.current_doc_id
            }
            for doc_id, meta in self.metadata.items()
        ]
    
    def switch_document(self, doc_id: str) -> bool:
        """Switch to a different loaded document"""
        if doc_id in self.documents:
            self.current_doc_id = doc_id
            logger.info(f"Switched to document: {doc_id}")
            return True
        return False


# ================= Interactive CLI =================

def interactive_mode():
    """Enhanced interactive mode with rich features"""
    print("\n" + "=" * 90)
    print("🚀 Advanced PDF Question Answering System v3.0")
    print("💡 AI-Powered Document Intelligence with Smart Fallback")
    print("=" * 90)
    
    # Initialize system
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("\n⚠️  No GEMINI_API_KEY found in environment")
        print("📝 To enable AI features, create a .env file with: GEMINI_API_KEY=your_key_here")
        print("🔍 Continuing in Traditional Search mode...\n")
    
    qa = AdvancedPDFQA(api_key=api_key, use_ai=True)
    
    # Status message
    if qa.ai_provider.available:
        print(f"✨ AI Mode: ENABLED ({qa.ai_provider.provider_name})")
    else:
        print("🔍 AI Mode: DISABLED (Traditional search only)")
    
    # Load default PDF
    default_pdf = "data/UML.pdf"
    if os.path.exists(default_pdf):
        print(f"\n📂 Loading default file: {default_pdf}")
        qa.load_pdf(default_pdf)
    else:
        print("\n📂 No default PDF found. Use 'load <path>' to load a document.")
    
    # Command help
    print("\n" + "=" * 90)
    print("📋 COMMANDS:")
    print("  💬 [question]          - Ask any question about the document")
    print("  🔍 search:<keyword>    - Search for specific keywords")
    print("  📄 summary             - Generate document summary")
    print("  📊 stats               - Show system statistics")
    print("  🔄 toggle              - Switch AI/Traditional mode")
    print("  📁 load <path>         - Load a PDF file")
    print("  📚 docs                - List loaded documents")
    print("  💾 cache               - Show cache statistics")
    print("  ❓ help                - Show detailed help")
    print("  🚪 exit                - Quit application")
    print("=" * 90)
    
    # Main loop
    while True:
        try:
            user_input = input("\n💬 ").strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.lower() == "exit":
                stats = qa.get_stats()
                print("\n" + "=" * 60)
                print("📊 SESSION SUMMARY")
                print("=" * 60)
                print(f"⏱️  Session Duration: {stats['session']['uptime']}")
                print(f"❓ Questions Asked: {stats['session']['questions_asked']}")
                print(f"🤖 AI Responses: {stats['session']['ai_responses']}")
                print(f"🔍 Traditional: {stats['session']['traditional_responses']}")
                print(f"💾 Cached: {stats['session']['cached_responses']}")
                print("=" * 60)
                print("👋 Thank you for using Advanced PDF QA System v3.0!")
                break
            
            elif user_input.lower() == "help":
                print("\n" + "=" * 70)
                print("📖 DETAILED HELP")
                print("=" * 70)
                print("\n💬 ASKING QUESTIONS:")
                print("  • Type naturally: 'What is UML?', 'Explain class diagrams'")
                print("  • AI provides intelligent, context-aware answers")
                print("  • Confidence scores help assess answer quality")
                print("  • Automatic fallback to traditional search if AI fails")
                print("\n🔍 SEARCH FEATURES:")
                print("  • search:keyword - Find all occurrences with context")
                print("  • Shows page numbers and surrounding text")
                print("  • Case-insensitive matching")
                print("\n📄 DOCUMENT MANAGEMENT:")
                print("  • load <path> - Load new PDF (supports up to 100MB)")
                print("  • docs - List all loaded documents")
                print("  • Multiple documents can be loaded simultaneously")
                print("\n⚙️ SYSTEM FEATURES:")
                print("  • Smart caching reduces API calls and improves speed")
                print("  • Rate limiting prevents API quota exhaustion")
                print("  • Confidence scoring for answer reliability")
                print("  • Comprehensive statistics and monitoring")
                print("=" * 70)
            
            elif user_input.lower() == "summary":
                print("\n📝 Generating summary...")
                summary = qa.get_summary()
                print(f"\n📄 SUMMARY:\n{summary}")
            
            elif user_input.lower() == "toggle":
                result = qa.toggle_ai_mode()
                print(f"\n{result}")
            
            elif user_input.lower() == "stats":
                stats = qa.get_stats()
                print("\n" + "=" * 70)
                print("📊 SYSTEM STATISTICS")
                print("=" * 70)
                print("\n📈 SESSION:")
                for key, value in stats['session'].items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
                print("\n📚 DOCUMENTS:")
                for key, value in stats['documents'].items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
                print("\n🤖 AI STATUS:")
                for key, value in stats['ai'].items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
                print("\n💾 CACHE PERFORMANCE:")
                print(f"  QA Cache: {stats['cache']['qa_cache']}")
                print(f"  Summary Cache: {stats['cache']['summary_cache']}")
                print("=" * 70)
            
            elif user_input.lower() == "docs":
                docs = qa.list_documents()
                if docs:
                    print("\n" + "=" * 70)
                    print("📚 LOADED DOCUMENTS")
                    print("=" * 70)
                    for doc in docs:
                        marker = "→" if doc['current'] else " "
                        print(f"{marker} {doc['filename']}")
                        print(f"  Pages: {doc['pages']} | Size: {doc['size']} | Loaded: {doc['loaded']}")
                        print()
                else:
                    print("\n📚 No documents loaded")
            
            elif user_input.lower() == "cache":
                stats = qa.get_stats()
                print("\n" + "=" * 60)
                print("💾 CACHE STATISTICS")
                print("=" * 60)
                print(f"\n🔍 QA Cache:")
                for key, value in stats['cache']['qa_cache'].items():
                    print(f"  {key.title()}: {value}")
                print(f"\n📄 Summary Cache:")
                for key, value in stats['cache']['summary_cache'].items():
                    print(f"  {key.title()}: {value}")
                print("=" * 60)
            
            elif user_input.lower().startswith("search:"):
                keyword = user_input[7:].strip()
                if keyword:
                    results = qa.search_keyword(keyword)
                    if results:
                        print(f"\n🔍 Found {len(results)} results for '{keyword}':")
                        print("=" * 70)
                        for i, result in enumerate(results[:10], 1):
                            print(f"\n{i}. Page {result['page']} (Line {result['line_number']})")
                            print(f"   {result['context'][:200]}...")
                    else:
                        print(f"\n❌ No results found for '{keyword}'")
                else:
                    print("\n❌ Please provide a keyword: search:<keyword>")
            
            elif user_input.lower().startswith("load "):
                path = user_input[5:].strip()
                qa.load_pdf(path)
            
            else:
                # Question answering
                print("\n🤔 Processing your question...")
                response = qa.answer_question(user_input)
                
                if 'error' in response:
                    print(f"\n❌ Error: {response['error']}")
                else:
                    print("\n" + "=" * 70)
                    print("✨ ANSWER")
                    print("=" * 70)
                    print(f"\n{response['answer']}")
                    print(f"\n📊 Confidence: {response['confidence']:.0%} | Mode: {response['mode']}")
                    if response.get('cached'):
                        print("💾 (Retrieved from cache)")
                    print("=" * 70)
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    interactive_mode()
