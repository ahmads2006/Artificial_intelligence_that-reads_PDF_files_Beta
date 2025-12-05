<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\PDFQAController;

Route::get('/test', function () {
    return view('test');
});


Route::post('/pdfqa/ask', [PDFQAController::class, 'ask']); // قم بتغيير الرابط إلى /pdfqa/ask  لتجنب التعارض مع الرابط الأصلي /ai/ask  