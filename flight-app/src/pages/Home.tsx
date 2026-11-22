import { Nav } from "../components/Nav";
import { SearchForm } from "../components/SearchForm";

export function Home() {
  return (
    <div className="app-shell">
      <Nav />
      <section className="hero" aria-label="Halcyon flight search">
        <div className="hero-sky" aria-hidden="true" />
        <svg
          className="plane-path"
          viewBox="0 0 1200 300"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <path d="M40,210 C220,40 420,40 600,150 C780,260 980,260 1160,90" />
        </svg>
        <div className="plane-marker" aria-hidden="true">
          <svg viewBox="0 0 24 24">
            <path d="M21 16v-2l-8-5V3.5a1.5 1.5 0 0 0-3 0V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z" />
          </svg>
        </div>

        <div className="hero-content">
          <h1 className="hero-brand">
            HALCYON<span>.</span>
          </h1>
          <p className="hero-copy">
            Clear skies, clear fares. Search sample routes and reserve a seat in
            minutes.
          </p>
          <SearchForm />
        </div>
      </section>
      <footer className="site-footer">
        Sample flight data for demonstration — not a live booking system.
      </footer>
    </div>
  );
}
