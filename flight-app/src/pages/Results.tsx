import { useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { FlightCard } from "../components/FlightCard";
import { Nav } from "../components/Nav";
import { SearchForm } from "../components/SearchForm";
import { getAirport } from "../data/airports";
import { searchFlights } from "../data/flights";
import type { CabinClass, Flight, SearchParams } from "../types";

type SortKey = "price" | "duration" | "depart";
type StopFilter = "any" | "nonstop" | "one-stop";

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

export function Results() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const search = useMemo(() => parseSearch(params), [params]);
  const [sort, setSort] = useState<SortKey>("price");
  const [stops, setStops] = useState<StopFilter>("any");

  const flights = useMemo(() => {
    let list = searchFlights(search);
    if (stops === "nonstop") list = list.filter((f) => f.stops === 0);
    if (stops === "one-stop") list = list.filter((f) => f.stops === 1);

    return [...list].sort((a, b) => {
      if (sort === "duration") return a.durationMinutes - b.durationMinutes;
      if (sort === "depart") return a.departTime.localeCompare(b.departTime);
      return a.price - b.price;
    });
  }, [search, sort, stops]);

  const fromCity = getAirport(search.from)?.city ?? search.from;
  const toCity = getAirport(search.to)?.city ?? search.to;

  function onSelect(flight: Flight) {
    const next = new URLSearchParams({
      flightId: flight.id,
      from: search.from,
      to: search.to,
      date: search.date,
      passengers: String(search.passengers),
      cabin: search.cabin,
      tripType: search.tripType,
    });
    navigate(`/book?${next.toString()}`);
  }

  return (
    <div className="app-shell">
      <Nav solid />
      <main className="page">
        <div className="page-header">
          <div>
            <h1>
              {fromCity} → {toCity}
            </h1>
            <p>
              {search.date} · {search.passengers} traveler
              {search.passengers > 1 ? "s" : ""} · {search.cabin}
              {search.tripType === "round-trip" ? " · round trip (outbound)" : ""}
            </p>
          </div>
        </div>

        <SearchForm initial={search} compact />

        <div className="results-layout" style={{ marginTop: "1.5rem" }}>
          <aside className="filters">
            <h2>Sort</h2>
            <div className="filter-group">
              {(
                [
                  ["price", "Lowest price"],
                  ["duration", "Shortest duration"],
                  ["depart", "Earliest departure"],
                ] as const
              ).map(([value, label]) => (
                <label key={value}>
                  <input
                    type="radio"
                    name="sort"
                    checked={sort === value}
                    onChange={() => setSort(value)}
                  />
                  {label}
                </label>
              ))}
            </div>

            <h2>Stops</h2>
            <div className="filter-group">
              {(
                [
                  ["any", "Any"],
                  ["nonstop", "Nonstop"],
                  ["one-stop", "1 stop"],
                ] as const
              ).map(([value, label]) => (
                <label key={value}>
                  <input
                    type="radio"
                    name="stops"
                    checked={stops === value}
                    onChange={() => setStops(value)}
                  />
                  {label}
                </label>
              ))}
            </div>
          </aside>

          <section className="flight-list" aria-live="polite">
            {flights.length === 0 ? (
              <div className="empty-state">
                <p>No sample flights match this route and cabin.</p>
                <p>Try JFK → LAX, SFO → SEA, or BOS → LHR.</p>
              </div>
            ) : (
              flights.map((flight) => (
                <FlightCard
                  key={flight.id}
                  flight={flight}
                  passengers={search.passengers}
                  onSelect={onSelect}
                />
              ))
            )}
          </section>
        </div>
      </main>
      <footer className="site-footer">
        Showing sample inventory — prices are illustrative.
      </footer>
    </div>
  );
}
