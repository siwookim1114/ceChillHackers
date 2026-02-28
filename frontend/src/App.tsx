import { Navigate, Route, Routes } from "react-router-dom";
import { CommunityPage } from "./pages/CommunityPage";
import { CreateCoursePage } from "./pages/CreateCoursePage";
import { EntryPage } from "./pages/EntryPage";
import { FriendsPage } from "./pages/FriendsPage";
import { HomePage } from "./pages/HomePage";
import { LibraryPage } from "./pages/LibraryPage";
import { LoginPage } from "./pages/LoginPage";
import { OnboardingPage } from "./pages/OnboardingPage";
import { PlannerPage } from "./pages/PlannerPage";
import { PracticePage } from "./pages/PracticePage";
import { ProgressPage } from "./pages/ProgressPage";
import { ReportBugPage } from "./pages/ReportBugPage";
import { ResultPage } from "./pages/ResultPage";
import { SavedPage } from "./pages/SavedPage";
import { SettingsPage } from "./pages/SettingsPage";
import { SolvePage } from "./pages/SolvePage";
import { StuffPage } from "./pages/StuffPage";
import { SignupPage } from "./pages/SignupPage";
import { SupportPage } from "./pages/SupportPage";
import { UpgradePage } from "./pages/UpgradePage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<EntryPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignupPage />} />
      <Route path="/onboarding" element={<OnboardingPage />} />
      <Route path="/home" element={<HomePage />} />
      <Route path="/practice" element={<PracticePage />} />
      <Route path="/create-course" element={<CreateCoursePage />} />
      <Route path="/planner" element={<PlannerPage />} />
      <Route path="/progress" element={<ProgressPage />} />
      <Route path="/library" element={<LibraryPage />} />
      <Route path="/community" element={<CommunityPage />} />
      <Route path="/settings" element={<SettingsPage />} />
      <Route path="/friends" element={<FriendsPage />} />
      <Route path="/saved" element={<SavedPage />} />
      <Route path="/stuff" element={<StuffPage />} />
      <Route path="/support" element={<SupportPage />} />
      <Route path="/report-bug" element={<ReportBugPage />} />
      <Route path="/upgrade" element={<UpgradePage />} />
      <Route path="/solve/:attemptId" element={<SolvePage />} />
      <Route path="/result/:attemptId" element={<ResultPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
