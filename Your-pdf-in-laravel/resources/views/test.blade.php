<h3>اختبار نظام الذكاء الاصطناعي للـ PDF</h3>

<div style="margin-bottom: 20px;">
    <label for="pdf_path">مسار ملف PDF:</label><br>
    <input type="text" id="pdf_path" placeholder="UML.pdf" style="width: 100%; padding: 8px; margin: 5px 0;" value="UML.pdf">
    <small style="color: #666;">يمكنك استخدام اسم الملف فقط (مثل: UML.pdf أو rules.pdf) أو المسار الكامل لملف PDF</small>
</div>

<div style="margin-bottom: 20px;">
    <label for="question">السؤال:</label><br>
    <input type="text" id="question" placeholder="اكتب سؤالك هنا" style="width: 100%; padding: 8px; margin: 5px 0;">
</div>

<button id="askBtn" style="padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">اسأل الذكاء الاصطناعي</button>

<div id="answer" style="margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 4px; min-height: 100px;"></div>

<div id="loading" style="display: none; color: #007bff; margin-top: 10px;">جاري المعالجة...</div>
<div id="error" style="display: none; color: #dc3545; margin-top: 10px;"></div>

<script>
document.getElementById('askBtn').addEventListener('click', async () => {
    const question = document.getElementById('question').value.trim();
    const pdf_path = document.getElementById('pdf_path').value.trim();

    // التحقق من الحقول المطلوبة
    if (!question || !pdf_path) {
        showError('يرجى ملء جميع الحقول المطلوبة');
        return;
    }

    // إظهار حالة التحميل
    showLoading(true);
    hideError();

    try {
        const response = await fetch('/pdfqa/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-TOKEN': '{{ csrf_token() }}'
            },
            body: JSON.stringify({ question, pdf_path })
        });

        const data = await response.json();

        if (data.error) {
            showError(data.error);
            document.getElementById('answer').innerText = '';
        } else {
            document.getElementById('answer').innerText = data.answer || 'لم يتم العثور على إجابة';
            hideError();
        }
    } catch (error) {
        showError('حدث خطأ في الاتصال: ' + error.message);
        document.getElementById('answer').innerText = '';
    } finally {
        showLoading(false);
    }
});

function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
    document.getElementById('askBtn').disabled = show;
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.innerText = message;
    errorDiv.style.display = 'block';
}

function hideError() {
    document.getElementById('error').style.display = 'none';
}
</script>
