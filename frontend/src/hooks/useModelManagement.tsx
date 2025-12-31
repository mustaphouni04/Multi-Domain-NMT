
import { useState, useEffect } from "react";
import { toast } from "@/hooks/use-toast";
import { getModelStatus, switchModel, ModelStatus } from "@/services/modelService";

export const useModelManagement = () => {
  const [modelStatus, setModelStatus] = useState<ModelStatus>({
    availableModels: [],
    currentModel: 'Helsinki-NLP/opus-mt-en-es',
    isUsingCustomModel: false
  });
  const [isLoading, setIsLoading] = useState(false);

  const loadModelStatus = async () => {
    setIsLoading(true);
    try {
      const status = await getModelStatus();
      setModelStatus(status);
    } catch (error) {
      console.error('Failed to load model status:', error);
      toast({
        description: "Failed to load model status",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSwitchModel = async (modelName: string) => {
    setIsLoading(true);
    try {
      const success = await switchModel(modelName);
      if (success) {
        await loadModelStatus();
        toast({
          description: `Switched to ${modelName}`,
        });
      } else {
        throw new Error('Failed to switch model');
      }
    } catch (error) {
      console.error('Model switch failed:', error);
      toast({
        description: "Failed to switch model",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadModelStatus();
  }, []);

  return {
    modelStatus,
    isLoading,
    loadModelStatus,
    handleSwitchModel
  };
};
