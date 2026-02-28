import { Navigate, Route, Routes } from "react-router-dom";
import { EntryPage } from "./pages/EntryPage";
import { HomePage } from "./pages/HomePage";
import { OnboardingPage } from "./pages/OnboardingPage";
import { ResultPage } from "./pages/ResultPage";
import { SolvePage } from "./pages/SolvePage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<EntryPage />} />
      <Route path="/onboarding" element={<OnboardingPage />} />
      <Route path="/home" element={<HomePage />} />
      <Route path="/solve/:attemptId" element={<SolvePage />} />
      <Route path="/result/:attemptId" element={<ResultPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
