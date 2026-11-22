export type CabinClass = "economy" | "premium" | "business";

export type Airport = {
  code: string;
  city: string;
  name: string;
};

export type Flight = {
  id: string;
  airline: string;
  flightNumber: string;
  from: string;
  to: string;
  departTime: string;
  arriveTime: string;
  durationMinutes: number;
  stops: number;
  price: number;
  cabin: CabinClass;
  seatsLeft: number;
};

export type SearchParams = {
  from: string;
  to: string;
  date: string;
  passengers: number;
  cabin: CabinClass;
  tripType: "one-way" | "round-trip";
};

export type Booking = {
  flight: Flight;
  search: SearchParams;
  passengerName: string;
  email: string;
  confirmationCode: string;
};
