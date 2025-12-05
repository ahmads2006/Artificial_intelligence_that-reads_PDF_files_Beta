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
        $this->pythonScript = base_path('python_ai/Ai-project-all/main.py'); // مسار ملف بايثون
    }

    public function ask(Request $request)
    {
        try {
            $request->validate([
                'question' => 'required|string|max:2000',
                'pdf_path' => 'required|string'
            ]);

            $question = $request->input('question');
            $pdfPath = $request->input('pdf_path');

            // تطبيع مسار PDF - إذا كان مسار نسبي، اجعله مطلق من مجلد البيانات
            if (!str_starts_with($pdfPath, '/')) {
                // مسار نسبي - ابحث في مجلد البيانات
                $dataDir = base_path('python_ai/Ai-project-all/data');
                $pdfPath = $dataDir . '/' . $pdfPath;
            }

            // تحقق من وجود ملف Python
            if (!file_exists($this->pythonScript)) {
                return response()->json([
                    'error' => 'Python script not found: ' . $this->pythonScript,
                    'answer' => '',
                    'confidence' => 0.0
                ]);
            }

            // تحقق من وجود ملف PDF
            if (!file_exists($pdfPath)) {
                return response()->json([
                    'error' => 'PDF file not found: ' . $pdfPath,
                    'answer' => '',
                    'confidence' => 0.0
                ]);
            }

            // استخدم python من البيئة للتوافق مع WSL
            $process = new Process(['python3', $this->pythonScript, $pdfPath, $question]);

            $process->setWorkingDirectory(dirname($this->pythonScript)); // تعيين مجلد العمل
            $process->setTimeout(60); // زيادة المهلة الزمنية

            // تمرير متغيرات البيئة المطلوبة
            $env = $_ENV;
            // إذا كان GEMINI_API_KEY موجود في Laravel .env
            if (env('GEMINI_API_KEY')) {
                $env['GEMINI_API_KEY'] = env('GEMINI_API_KEY');
            }
            $process->setEnv($env);

            $process->run();

            if (!$process->isSuccessful()) {
                $errorOutput = $process->getErrorOutput();
                return response()->json([
                    'error' => 'Process failed: ' . $errorOutput,
                    'exit_code' => $process->getExitCode(),
                    'answer' => '',
                    'confidence' => 0.0
                ]);
            }

            $output = $process->getOutput();

            // تحقق من أن الإخراج ليس فارغاً
            if (empty($output)) {
                return response()->json([
                    'error' => 'No output from Python script',
                    'answer' => '',
                    'confidence' => 0.0
                ]);
            }

            $response = json_decode($output, true);

            // تحقق من صحة JSON
            if (json_last_error() !== JSON_ERROR_NONE) {
                return response()->json([
                    'error' => 'Invalid JSON response: ' . json_last_error_msg(),
                    'raw_output' => $output,
                    'answer' => '',
                    'confidence' => 0.0
                ]);
            }

            return response()->json($response);

        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Exception: ' . $e->getMessage(),
                'answer' => '',
                'confidence' => 0.0
            ]);
        }
    }
}
