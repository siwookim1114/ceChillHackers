import { Navigate, Route, Routes } from "react-router-dom";
import { CommunityPage } from "./pages/CommunityPage";
import { EntryPage } from "./pages/EntryPage";
import { HomePage } from "./pages/HomePage";
import { LibraryPage } from "./pages/LibraryPage";
import { LoginPage } from "./pages/LoginPage";
import { OnboardingPage } from "./pages/OnboardingPage";
import { PlannerPage } from "./pages/PlannerPage";
import { PracticePage } from "./pages/PracticePage";
import { ProgressPage } from "./pages/ProgressPage";
import { ResultPage } from "./pages/ResultPage";
import { SettingsPage } from "./pages/SettingsPage";
import { SolvePage } from "./pages/SolvePage";
import { SignupPage } from "./pages/SignupPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<EntryPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignupPage />} />
      <Route path="/onboarding" element={<OnboardingPage />} />
      <Route path="/home" element={<HomePage />} />
      <Route path="/practice" element={<PracticePage />} />
      <Route path="/planner" element={<PlannerPage />} />
      <Route path="/progress" element={<ProgressPage />} />
      <Route path="/library" element={<LibraryPage />} />
      <Route path="/community" element={<CommunityPage />} />
      <Route path="/settings" element={<SettingsPage />} />
      <Route path="/solve/:attemptId" element={<SolvePage />} />
      <Route path="/result/:attemptId" element={<ResultPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
