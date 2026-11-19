package com.waeil.commence.mcp;

import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.ai.tool.method.MethodToolCallbackProvider;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class McpToolConfig {

    @Bean
    public ToolCallbackProvider cartToolCallbackProvider(CartTools cartTools) {
        return MethodToolCallbackProvider.builder()
                .toolObjects(cartTools)
                .build();
    }
}