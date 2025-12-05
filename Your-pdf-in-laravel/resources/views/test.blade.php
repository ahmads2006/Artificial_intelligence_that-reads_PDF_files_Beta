<input type="text" id="question" placeholder="اكتب سؤالك">
<input type="text" id="pdf_path" placeholder="مسار PDF">
<button id="askBtn">اسأل الذكاء الصناعي</button>

<div id="answer"></div>

<script>
document.getElementById('askBtn').addEventListener('click', async () => {
    const question = document.getElementById('question').value;
    const pdf_path = document.getElementById('pdf_path').value;

    const response = await fetch('/api/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRF-TOKEN': '{{ csrf_token() }}'
        },
        body: JSON.stringify({ question, pdf_path })
    });

    const data = await response.json();
    document.getElementById('answer').innerText = data.answer;
});
</script>
