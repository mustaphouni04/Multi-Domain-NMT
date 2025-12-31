
export interface TranslationRequest {
  text: string;
  source_lang: string;
  target_lang: string;
}

export interface TranslationResponse {
  translated_text: string;
  confidence?: number;
}

export interface MqxliffTranslationRequest {
  fileContent: string;
  fileName: string;
  sourceLang: string;
  targetLang: string;
}

export interface MqxliffTranslationResponse {
  success: boolean;
  translatedContent?: string;
  error?: string;
}

export const translateText = async (request: TranslationRequest): Promise<TranslationResponse> => {
  const response = await fetch('/api/translate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request)
  });

  if (!response.ok) {
    throw new Error('Translation failed');
  }

  return response.json();
};

export const translateMqxliff = async (request: MqxliffTranslationRequest): Promise<MqxliffTranslationResponse> => {
  const response = await fetch('/api/translate/mqxliff', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request)
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ error: 'Translation failed' }));
    return {
      success: false,
      error: errorData.error || 'Translation failed'
    };
  }

  const data = await response.json();
  return {
    success: true,
    translatedContent: data.translated_content
  };
};
