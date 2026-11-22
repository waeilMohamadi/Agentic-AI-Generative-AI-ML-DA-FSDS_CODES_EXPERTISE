import { formatDuration, formatPrice } from "../data/flights";
import { getAirport } from "../data/airports";
import type { Flight } from "../types";

type FlightCardProps = {
  flight: Flight;
  passengers: number;
  onSelect: (flight: Flight) => void;
};

export function FlightCard({ flight, passengers, onSelect }: FlightCardProps) {
  const from = getAirport(flight.from);
  const to = getAirport(flight.to);
  const total = flight.price * passengers;

  return (
    <button
      type="button"
      className="flight-option"
      onClick={() => onSelect(flight)}
      aria-label={`Select ${flight.airline} ${flight.flightNumber} for ${formatPrice(total)}`}
    >
      <div>
        <div className="flight-meta">
          <span>{flight.airline}</span>
          <span>{flight.flightNumber}</span>
          <span>{flight.stops === 0 ? "Nonstop" : `${flight.stops} stop`}</span>
          <span>{flight.seatsLeft} seats left</span>
        </div>
        <div className="flight-route">
          <div className="time-block">
            <strong>{flight.departTime}</strong>
            <span>
              {from?.city} ({flight.from})
            </span>
          </div>
          <div className="route-line">
            <em>{formatDuration(flight.durationMinutes)}</em>
          </div>
          <div className="time-block">
            <strong>{flight.arriveTime}</strong>
            <span>
              {to?.city} ({flight.to})
            </span>
          </div>
        </div>
      </div>
      <div className="flight-price">
        <strong>{formatPrice(total)}</strong>
        <span>
          {passengers > 1
            ? `${formatPrice(flight.price)} × ${passengers}`
            : "per traveler"}
        </span>
      </div>
    </button>
  );
}
