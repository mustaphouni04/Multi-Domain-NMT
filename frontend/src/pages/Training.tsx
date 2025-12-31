
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from "react-router-dom";
import { FileText, Home, Play, Square, BarChart3, Settings } from "lucide-react";
import { useState } from "react";

const Training = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);

  const handleStartTraining = () => {
    setIsTraining(true);
    // Simulate training progress
    const interval = setInterval(() => {
      setTrainingProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          return 0;
        }
        return prev + 5;
      });
    }, 200);
  };

  const handleStopTraining = () => {
    setIsTraining(false);
    setTrainingProgress(0);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-red-600 mb-2">
            Model Training
          </h1>
          <p className="text-gray-600 max-w-2xl mx-auto mb-6">
            Train custom neural machine translation models for specific domains and use cases.
          </p>
          
          {/* Navigation */}
          <div className="flex justify-center gap-4 mb-8">
            <Link to="/">
              <Button variant="outline" className="flex items-center gap-2">
                <Home className="h-4 w-4" />
                Text Translation
              </Button>
            </Link>
            <Link to="/mqxliff">
              <Button variant="outline" className="flex items-center gap-2">
                <FileText className="h-4 w-4" />
                MQXLIFF Translation
              </Button>
            </Link>
            <Button variant="default" className="bg-red-600 hover:bg-red-700">
              Model Training
            </Button>
          </div>
        </header>

        {/* Main Content */}
        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Training Configuration */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Training Configuration
              </CardTitle>
              <CardDescription>
                Configure your model training parameters
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Training Type</label>
                  <select className="w-full p-2 border rounded-md">
                    <option value="maml">MAML</option>
                    <option value="standard">Standard Fine-tuning</option>
                    <option value="distillation">Knowledge Distillation</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Domain</label>
                  <select className="w-full p-2 border rounded-md">
                    <option value="medical">Medical</option>
                    <option value="legal">Legal</option>
                    <option value="technical">Technical</option>
                    <option value="general">General</option>
                  </select>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Batch Size</label>
                  <input type="number" defaultValue="12" className="w-full p-2 border rounded-md" />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Learning Rate</label>
                  <input type="number" step="0.0001" defaultValue="0.0003" className="w-full p-2 border rounded-md" />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Max Epochs</label>
                <input type="number" defaultValue="100" className="w-full p-2 border rounded-md" />
              </div>

              {/* Training Controls */}
              <div className="flex gap-4 pt-4">
                {!isTraining ? (
                  <Button 
                    onClick={handleStartTraining}
                    className="flex items-center gap-2 bg-green-600 hover:bg-green-700"
                  >
                    <Play className="h-4 w-4" />
                    Start Training
                  </Button>
                ) : (
                  <Button 
                    onClick={handleStopTraining}
                    variant="destructive"
                    className="flex items-center gap-2"
                  >
                    <Square className="h-4 w-4" />
                    Stop Training
                  </Button>
                )}
              </div>

              {/* Progress Bar */}
              {isTraining && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Training Progress</span>
                    <span>{trainingProgress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${trainingProgress}%` }}
                    ></div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Training Status & Metrics */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Training Status
              </CardTitle>
              <CardDescription>
                Monitor your training progress and metrics
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {isTraining ? 'Running' : 'Idle'}
                  </div>
                  <div className="text-sm text-gray-600">Current Status</div>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {isTraining ? Math.floor(trainingProgress / 10) : 0}
                  </div>
                  <div className="text-sm text-gray-600">Epochs Completed</div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="bg-purple-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">0.85</div>
                  <div className="text-sm text-gray-600">Best BLEU Score</div>
                </div>
                <div className="bg-orange-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-orange-600">2.3</div>
                  <div className="text-sm text-gray-600">Current Loss</div>
                </div>
              </div>

              {/* Recent Training Logs */}
              <div className="space-y-2">
                <h4 className="font-medium">Recent Logs</h4>
                <div className="bg-gray-50 p-3 rounded-md text-sm font-mono h-32 overflow-y-auto">
                  {isTraining ? (
                    <div className="space-y-1">
                      <div>[{new Date().toLocaleTimeString()}] Training started...</div>
                      <div>[{new Date().toLocaleTimeString()}] Loading training data...</div>
                      <div>[{new Date().toLocaleTimeString()}] Epoch 1 started</div>
                      <div>[{new Date().toLocaleTimeString()}] Batch processing...</div>
                    </div>
                  ) : (
                    <div className="text-gray-500">No active training session</div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Available Models Section */}
        <Card className="mt-8 max-w-6xl mx-auto">
          <CardHeader>
            <CardTitle>Available Models</CardTitle>
            <CardDescription>
              Pre-trained models ready for fine-tuning or inference
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="border rounded-lg p-4">
                <h4 className="font-medium">Helsinki EN-ES</h4>
                <p className="text-sm text-gray-600 mb-2">General English to Spanish translation</p>
                <Button size="sm" variant="outline">Load Model</Button>
              </div>
              <div className="border rounded-lg p-4">
                <h4 className="font-medium">Medical Domain</h4>
                <p className="text-sm text-gray-600 mb-2">Specialized for medical texts</p>
                <Button size="sm" variant="outline">Load Model</Button>
              </div>
              <div className="border rounded-lg p-4">
                <h4 className="font-medium">Legal Domain</h4>
                <p className="text-sm text-gray-600 mb-2">Optimized for legal documents</p>
                <Button size="sm" variant="outline">Load Model</Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Training;
