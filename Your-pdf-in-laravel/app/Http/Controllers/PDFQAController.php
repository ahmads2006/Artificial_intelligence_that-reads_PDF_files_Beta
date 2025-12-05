<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Symfony\Component\Process\Process;
use Symfony\Component\Process\Exception\ProcessFailedException;

class PDFQAController extends Controller
{
    protected $pythonScript;

    public function __construct()
    {
        $this->pythonScript = base_path('python/pdf_qa.py'); // مسار ملف بايثون
    }

    public function ask(Request $request)
    {
        $request->validate([
            'question' => 'required|string|max:2000',
            'pdf_path' => 'required|string'
        ]);

        $question = $request->input('question');
        $pdfPath = $request->input('pdf_path');

        $process = new Process(['python3', $this->pythonScript, $pdfPath, $question]);
        $process->run();

        if (!$process->isSuccessful()) {
            throw new ProcessFailedException($process);
        }

        $output = $process->getOutput();
        $response = json_decode($output, true);

        return response()->json($response);
    }
}
