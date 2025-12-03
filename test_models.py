import google.generativeai as genai

# إعداد API
api_key = "AIzaSyCw0kuoadN9sKRjVF6VOZrYc2PCfwxsD2k"
genai.configure(api_key=api_key)

# قائمة النماذج المتاحة
print("النماذج المتاحة:")
try:
    models = genai.list_models()
    for model in models:
        print(f"- {model.name}: {model.description}")
        if hasattr(model, 'supported_generation_methods'):
            print(f"  دعم: {model.supported_generation_methods}")
        print()
except Exception as e:
    print(f"خطأ في قائمة النماذج: {e}")

# تجربة نماذج مختلفة
models_to_test = ['gemini-pro', 'gemini-1.0-pro', 'gemini-1.5-pro']

print("\nاختبار النماذج:")
for model_name in models_to_test:
    try:
        model = genai.GenerativeModel(model_name)
        print(f"✅ {model_name}: نجح")
    except Exception as e:
        print(f"❌ {model_name}: {e}")