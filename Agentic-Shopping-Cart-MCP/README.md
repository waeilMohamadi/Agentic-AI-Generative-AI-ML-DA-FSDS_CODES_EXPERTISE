# Agentic Shopping Cart Assistant

A Java 17 and Spring Boot shopping cart service that exposes cart actions as Model Context Protocol (MCP) tools. An AI assistant connected to the server can interpret a request, call the relevant tool, inspect the cart, and explain the result. The project demonstrates agentic tool use with a persistent MySQL cart; the model and chat interface belong to the external MCP client.

## What this demonstrates

- **Spring Boot and Spring AI:** a Java MCP server with annotated tools registered through a `ToolCallbackProvider`.
- **Agentic interaction:** the assistant can call `addToCart`, `removeQuantityFromCart`, `removeFromCart`, `listCart`, `listProducts`, and `getCartTotal` in response to a user request.
- **Data persistence:** Spring Data JPA stores cart entries in MySQL, with quantity and price recorded per line.
- **Accurate updates:** removing 20 units from a line with 100 units now leaves 80 directly; invalid or excessive removals do not change the cart.

## Technology

Java 17, Spring Boot 3.5.11, Spring AI 1.1.2, MCP over stdio, Spring Data JPA, MySQL, Maven, JUnit 5, and Mockito.

## Run locally

1. Install JDK 17, Maven, and MySQL. Create a database named `shopping_cart`.
2. Set `SHOPPING_CART_DB_URL`, `SHOPPING_CART_DB_USER`, and `SHOPPING_CART_DB_PASSWORD` in your local environment. The URL defaults to `jdbc:mysql://localhost:3306/shopping_cart`; the username defaults to `root`. Do not commit credentials.
3. From this directory, build with `mvn clean package` and run with `java -jar target/Agentic-Ai-System-With-Spring-Boot-0.0.1-SNAPSHOT.jar`.
4. Configure an MCP client to launch that Java command from this directory, then restart the client. The server uses stdio; its standard output is reserved for MCP messages. Pass database environment variables to the launched process if the client does not inherit them.

On Windows PowerShell, set environment variables for the current session before building or launching:

```powershell
$env:SHOPPING_CART_DB_URL = 'jdbc:mysql://localhost:3306/shopping_cart'
$env:SHOPPING_CART_DB_USER = 'root'
$env:SHOPPING_CART_DB_PASSWORD = 'your-local-password'
mvn clean package
```

For a demo, ask the connected assistant to add 100 iPhones, remove 20 iPhones, then list the cart. `removeQuantityFromCart` should leave 80. The older `removeFromCart` intentionally deletes the entire product line.

## Current scope

The cart is a single shared cart. Adding items uses the built-in demo catalog (iPhone, laptop, mouse). `listProducts` reads the separate `products` table, so it returns no registered products until that table is populated. The project has no authentication, inventory reservation, checkout, or multi-user cart support. Prices use `Double` for this demo; financial production systems should use decimal arithmetic.

The original workspace archive contained build outputs and machine-specific settings. This package includes only source, Maven configuration, documentation, and tests. The database password is supplied at runtime.
