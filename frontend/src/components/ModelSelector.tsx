
import { useModelManagement } from "@/hooks/useModelManagement";
import { Button } from "@/components/ui/button";
import { Settings } from "lucide-react";

export const ModelSelector = () => {
  const { modelStatus, isLoading, handleSwitchModel } = useModelManagement();

  return (
    <div className="mb-6 p-4 bg-white rounded-lg shadow-sm border">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Settings className="h-5 w-5 text-gray-600" />
          <span className="font-medium">Model:</span>
          <span className="text-sm text-gray-600">
            {modelStatus.currentModel}
            {modelStatus.isUsingCustomModel && " (Custom)"}
          </span>
        </div>
        
        {modelStatus.availableModels.length > 1 && (
          <div className="flex gap-2">
            {modelStatus.availableModels.map((model) => (
              <Button
                key={model.path}
                variant={model.path === modelStatus.currentModel ? "default" : "outline"}
                size="sm"
                disabled={isLoading}
                onClick={() => handleSwitchModel(model.path)}
                className="text-xs"
              >
                {model.isCustom ? "Custom" : "Base"}
              </Button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
