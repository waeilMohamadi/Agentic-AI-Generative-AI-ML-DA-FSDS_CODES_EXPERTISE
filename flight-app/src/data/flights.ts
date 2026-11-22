import type { CabinClass, Flight, SearchParams } from "../types";

const airlines = [
  "Halcyon Airways",
  "Pacific Drift",
  "Nordic Arc",
  "Coastal Jet",
  "Meridian Air",
];

const baseRoutes: Array<{
  from: string;
  to: string;
  hours: number;
  basePrice: number;
}> = [
  { from: "JFK", to: "LAX", hours: 6.2, basePrice: 289 },
  { from: "JFK", to: "LHR", hours: 7.5, basePrice: 520 },
  { from: "JFK", to: "MIA", hours: 3.1, basePrice: 198 },
  { from: "LAX", to: "NRT", hours: 11.4, basePrice: 740 },
  { from: "LAX", to: "SFO", hours: 1.4, basePrice: 119 },
  { from: "LAX", to: "SYD", hours: 15.2, basePrice: 980 },
  { from: "ORD", to: "DEN", hours: 2.6, basePrice: 165 },
  { from: "ORD", to: "CDG", hours: 8.3, basePrice: 610 },
  { from: "SFO", to: "SEA", hours: 2.1, basePrice: 142 },
  { from: "SFO", to: "DXB", hours: 15.8, basePrice: 890 },
  { from: "BOS", to: "LHR", hours: 6.8, basePrice: 455 },
  { from: "SEA", to: "NRT", hours: 10.5, basePrice: 680 },
  { from: "MIA", to: "CDG", hours: 9.1, basePrice: 540 },
  { from: "DEN", to: "ATL", hours: 2.9, basePrice: 178 },
  { from: "DFW", to: "LAX", hours: 3.4, basePrice: 210 },
  { from: "ATL", to: "JFK", hours: 2.3, basePrice: 156 },
  { from: "LHR", to: "CDG", hours: 1.3, basePrice: 98 },
  { from: "DXB", to: "SYD", hours: 13.6, basePrice: 820 },
];

function pad(n: number): string {
  return String(n).padStart(2, "0");
}

function formatTime(hours: number, minutes: number): string {
  return `${pad(hours)}:${pad(minutes)}`;
}

function addMinutes(startH: number, startM: number, durationMin: number) {
  const total = startH * 60 + startM + durationMin;
  const h = Math.floor(total / 60) % 24;
  const m = total % 60;
  return formatTime(h, m);
}

function cabinMultiplier(cabin: CabinClass): number {
  if (cabin === "premium") return 1.55;
  if (cabin === "business") return 2.8;
  return 1;
}

function buildCatalog(): Flight[] {
  const flights: Flight[] = [];
  let id = 1;

  for (const route of baseRoutes) {
    const reverse = {
      from: route.to,
      to: route.from,
      hours: route.hours,
      basePrice: route.basePrice,
    };
    for (const r of [route, reverse]) {
      for (let variant = 0; variant < 3; variant++) {
        const departH = 6 + variant * 5 + ((id * 3) % 4);
        const departM = (id * 7) % 60;
        const durationMinutes = Math.round(r.hours * 60) + variant * 25;
        const stops = variant === 2 ? 1 : 0;
        const airline = airlines[(id + variant) % airlines.length];
        const priceBump = variant * 35 + (stops === 0 ? 40 : -25);

        for (const cabin of ["economy", "premium", "business"] as CabinClass[]) {
          flights.push({
            id: `HX-${id}${cabin[0].toUpperCase()}${variant}`,
            airline,
            flightNumber: `HX${100 + id + variant}`,
            from: r.from,
            to: r.to,
            departTime: formatTime(departH % 24, departM),
            arriveTime: addMinutes(departH % 24, departM, durationMinutes),
            durationMinutes,
            stops,
            price: Math.round(
              (r.basePrice + priceBump) * cabinMultiplier(cabin),
            ),
            cabin,
            seatsLeft: 3 + ((id + variant) % 12),
          });
          id += 1;
        }
      }
    }
  }

  return flights;
}

export const flightCatalog = buildCatalog();

export function searchFlights(params: SearchParams): Flight[] {
  return flightCatalog
    .filter(
      (f) =>
        f.from === params.from &&
        f.to === params.to &&
        f.cabin === params.cabin,
    )
    .sort((a, b) => a.price - b.price);
}

export function formatDuration(minutes: number): string {
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return `${h}h ${m}m`;
}

export function formatPrice(price: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(price);
}
