import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Booking } from "./pages/Booking";
import { Home } from "./pages/Home";
import { Results } from "./pages/Results";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/flights" element={<Results />} />
        <Route path="/book" element={<Booking />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
