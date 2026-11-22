import { useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { airports } from "../data/airports";
import type { CabinClass, SearchParams } from "../types";

type SearchFormProps = {
  initial?: Partial<SearchParams>;
  compact?: boolean;
};

function defaultDate(): string {
  const d = new Date();
  d.setDate(d.getDate() + 14);
  return d.toISOString().slice(0, 10);
}

export function SearchForm({ initial, compact = false }: SearchFormProps) {
  const navigate = useNavigate();
  const [tripType, setTripType] = useState<"one-way" | "round-trip">(
    initial?.tripType ?? "one-way",
  );
  const [from, setFrom] = useState(initial?.from ?? "JFK");
  const [to, setTo] = useState(initial?.to ?? "LAX");
  const [date, setDate] = useState(initial?.date ?? defaultDate());
  const [passengers, setPassengers] = useState(initial?.passengers ?? 1);
  const [cabin, setCabin] = useState<CabinClass>(initial?.cabin ?? "economy");

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (from === to) return;

    const params = new URLSearchParams({
      from,
      to,
      date,
      passengers: String(passengers),
      cabin,
      tripType,
    });
    navigate(`/flights?${params.toString()}`);
  }

  return (
    <form className="search-panel" onSubmit={onSubmit} id="search">
      {!compact && (
        <div className="trip-toggle" role="group" aria-label="Trip type">
          <button
            type="button"
            className={tripType === "one-way" ? "active" : ""}
            onClick={() => setTripType("one-way")}
          >
            One way
          </button>
          <button
            type="button"
            className={tripType === "round-trip" ? "active" : ""}
            onClick={() => setTripType("round-trip")}
          >
            Round trip
          </button>
        </div>
      )}

      <div className="search-grid">
        <div className="field field--wide">
          <label htmlFor="from">From</label>
          <select
            id="from"
            value={from}
            onChange={(e) => setFrom(e.target.value)}
            required
          >
            {airports.map((airport) => (
              <option key={airport.code} value={airport.code}>
                {airport.city} ({airport.code})
              </option>
            ))}
          </select>
        </div>

        <div className="field field--wide">
          <label htmlFor="to">To</label>
          <select
            id="to"
            value={to}
            onChange={(e) => setTo(e.target.value)}
            required
          >
            {airports.map((airport) => (
              <option key={airport.code} value={airport.code}>
                {airport.city} ({airport.code})
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="date">Depart</label>
          <input
            id="date"
            type="date"
            value={date}
            min={new Date().toISOString().slice(0, 10)}
            onChange={(e) => setDate(e.target.value)}
            required
          />
        </div>

        <div className="field">
          <label htmlFor="passengers">Travelers</label>
          <select
            id="passengers"
            value={passengers}
            onChange={(e) => setPassengers(Number(e.target.value))}
          >
            {[1, 2, 3, 4, 5, 6].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="cabin">Cabin</label>
          <select
            id="cabin"
            value={cabin}
            onChange={(e) => setCabin(e.target.value as CabinClass)}
          >
            <option value="economy">Economy</option>
            <option value="premium">Premium</option>
            <option value="business">Business</option>
          </select>
        </div>

        <div className="field field--action">
          <label htmlFor="search-submit">&nbsp;</label>
          <button id="search-submit" className="btn btn-primary" type="submit">
            Search flights
          </button>
        </div>
      </div>
    </form>
  );
}
