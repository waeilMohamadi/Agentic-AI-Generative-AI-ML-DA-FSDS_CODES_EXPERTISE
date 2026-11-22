import { Link } from "react-router-dom";

type NavProps = {
  solid?: boolean;
};

export function Nav({ solid = false }: NavProps) {
  return (
    <header className={`site-nav${solid ? " site-nav--solid" : ""}`}>
      <Link to="/" className="brand">
        HALCYON<span>.</span>
      </Link>
      <nav className="nav-links" aria-label="Primary">
        <a href="#search">Search</a>
        <Link to="/?from=JFK&to=LAX">Popular routes</Link>
      </nav>
    </header>
  );
}
