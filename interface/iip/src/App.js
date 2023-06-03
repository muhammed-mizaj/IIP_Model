import logo from "./logo.svg";
import "./App.css";
import Home from "./pages/Home";
import { BrowserRouter, Routes, Route } from "react-router-dom";

function App() {
  return (
    <div className="App">
      <h1 className="bg-indigo-700">Intelligent Invoice Parser</h1>

      <BrowserRouter>
        <Routes>
          <Route path={"/"} element={<Home />} exact />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
