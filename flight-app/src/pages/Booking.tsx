import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Nav } from "../components/Nav";
import { getAirport } from "../data/airports";
import { flightCatalog, formatDuration, formatPrice } from "../data/flights";
import type { CabinClass, SearchParams } from "../types";

function parseSearch(params: URLSearchParams): SearchParams {
  return {
    from: params.get("from") ?? "JFK",
    to: params.get("to") ?? "LAX",
    date: params.get("date") ?? new Date().toISOString().slice(0, 10),
    passengers: Number(params.get("passengers") ?? 1),
    cabin: (params.get("cabin") as CabinClass) || "economy",
    tripType:
      params.get("tripType") === "round-trip" ? "round-trip" : "one-way",
  };
}

function makeConfirmationCode(): string {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let code = "";
  for (let i = 0; i < 6; i += 1) {
    code += alphabet[Math.floor(Math.random() * alphabet.length)];
  }
  return code;
}

export function Booking() {
  const [params] = useSearchParams();
  const search = useMemo(() => parseSearch(params), [params]);
  const flightId = params.get("flightId");
  const flight = flightCatalog.find((f) => f.id === flightId);

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [confirmed, setConfirmed] = useState<string | null>(null);

  if (!flight) {
    return (
      <div className="app-shell">
        <Nav solid />
        <main className="page">
          <div className="empty-state">
            <p>That flight could not be found.</p>
            <Link className="btn btn-primary" to="/" style={{ marginTop: "1rem" }}>
              Back to search
            </Link>
          </div>
        </main>
      </div>
    );
  }

  const from = getAirport(flight.from);
  const to = getAirport(flight.to);
  const total = flight.price * search.passengers;

  function onConfirm(event: FormEvent) {
    event.preventDefault();
    if (!name.trim() || !email.trim()) return;

    const code = makeConfirmationCode();
    const booking = {
      flight,
      search,
      passengerName: name.trim(),
      email: email.trim(),
      confirmationCode: code,
    };
    localStorage.setItem("halcyon-last-booking", JSON.stringify(booking));
    setConfirmed(code);
  }

  if (confirmed) {
    return (
      <div className="app-shell">
        <Nav solid />
        <main className="page">
          <div className="confirmation panel">
            <h1>You&apos;re booked</h1>
            <p>
              {name}, your sample reservation on {flight.airline}{" "}
              {flight.flightNumber} is held locally in this browser.
            </p>
            <div className="confirmation-code">{confirmed}</div>
            <p>
              {from?.city} ({flight.from}) → {to?.city} ({flight.to}) on{" "}
              {search.date} at {flight.departTime}
            </p>
            <div style={{ marginTop: "1.5rem", display: "flex", gap: "0.75rem", justifyContent: "center" }}>
              <Link className="btn btn-primary" to="/">
                Book another flight
              </Link>
              <Link
                className="btn btn-ghost"
                to={`/flights?from=${search.from}&to=${search.to}&date=${search.date}&passengers=${search.passengers}&cabin=${search.cabin}&tripType=${search.tripType}`}
              >
                Back to results
              </Link>
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <Nav solid />
      <main className="page">
        <div className="page-header">
          <div>
            <h1>Passenger details</h1>
            <p>
              Complete your sample booking for {flight.airline}{" "}
              {flight.flightNumber}.
            </p>
          </div>
        </div>

        <div className="booking-grid">
          <form className="panel form-stack" onSubmit={onConfirm}>
            <h2>Traveler</h2>
            <div className="field">
              <label htmlFor="name">Full name</label>
              <input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Alex Traveler"
                required
                autoComplete="name"
              />
            </div>
            <div className="field">
              <label htmlFor="email">Email</label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="alex@example.com"
                required
                autoComplete="email"
              />
            </div>
            <button className="btn btn-primary" type="submit">
              Confirm booking · {formatPrice(total)}
            </button>
            <p className="sample-note">
              This is a demo checkout. No payment is collected and nothing is
              sent to an airline.
            </p>
          </form>

          <aside className="panel">
            <h2>Trip summary</h2>
            <div className="summary-row">
              <span>Route</span>
              <span>
                {flight.from} → {flight.to}
              </span>
            </div>
            <div className="summary-row">
              <span>Date</span>
              <span>{search.date}</span>
            </div>
            <div className="summary-row">
              <span>Depart / Arrive</span>
              <span>
                {flight.departTime} → {flight.arriveTime}
              </span>
            </div>
            <div className="summary-row">
              <span>Duration</span>
              <span>{formatDuration(flight.durationMinutes)}</span>
            </div>
            <div className="summary-row">
              <span>Cabin</span>
              <span>{flight.cabin}</span>
            </div>
            <div className="summary-row">
              <span>Travelers</span>
              <span>{search.passengers}</span>
            </div>
            <div className="summary-row">
              <span>Total</span>
              <span>{formatPrice(total)}</span>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}
