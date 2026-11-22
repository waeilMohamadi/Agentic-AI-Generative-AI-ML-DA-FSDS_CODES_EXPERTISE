import type { Airport } from "../types";

export const airports: Airport[] = [
  { code: "JFK", city: "New York", name: "John F. Kennedy International" },
  { code: "LAX", city: "Los Angeles", name: "Los Angeles International" },
  { code: "ORD", city: "Chicago", name: "O'Hare International" },
  { code: "SFO", city: "San Francisco", name: "San Francisco International" },
  { code: "MIA", city: "Miami", name: "Miami International" },
  { code: "SEA", city: "Seattle", name: "Seattle-Tacoma International" },
  { code: "BOS", city: "Boston", name: "Logan International" },
  { code: "DEN", city: "Denver", name: "Denver International" },
  { code: "ATL", city: "Atlanta", name: "Hartsfield-Jackson Atlanta" },
  { code: "DFW", city: "Dallas", name: "Dallas/Fort Worth International" },
  { code: "LHR", city: "London", name: "Heathrow" },
  { code: "CDG", city: "Paris", name: "Charles de Gaulle" },
  { code: "NRT", city: "Tokyo", name: "Narita International" },
  { code: "DXB", city: "Dubai", name: "Dubai International" },
  { code: "SYD", city: "Sydney", name: "Kingsford Smith" },
];

export function getAirport(code: string): Airport | undefined {
  return airports.find((a) => a.code === code);
}
