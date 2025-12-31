
export interface ModelInfo {
  path: string;
  isCustom: boolean;
}

export interface ModelStatus {
  availableModels: ModelInfo[];
  currentModel: string;
  isUsingCustomModel: boolean;
}

export const getModelStatus = async (): Promise<ModelStatus> => {
  const response = await fetch('/api/models');
  
  if (!response.ok) {
    throw new Error('Failed to fetch model status');
  }
  
  const data = await response.json();
  
  // Transform the response to match our interface
  const availableModels: ModelInfo[] = data.models.map((model: any) => ({
    path: model.id,
    isCustom: model.type === 'domain'
  }));
  
  // For now, we'll assume the first model is current (you may want to enhance this)
  const currentModel = availableModels.length > 0 ? availableModels[0].path : 'Helsinki-NLP/opus-mt-en-es';
  
  return {
    availableModels,
    currentModel,
    isUsingCustomModel: availableModels.some(m => m.isCustom && m.path === currentModel)
  };
};

export const switchModel = async (modelName: string): Promise<boolean> => {
  const response = await fetch('/api/models/load', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ model_name: modelName })
  });
  
  if (!response.ok) {
    throw new Error('Failed to switch model');
  }
  
  const data = await response.json();
  return data.message ? true : false;
};
