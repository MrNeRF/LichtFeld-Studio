#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D leftTexture;
uniform sampler2D rightTexture;
uniform float splitPosition;
uniform bool showDivider;
uniform vec4 dividerColor;
uniform float dividerWidth;
uniform vec2 leftTexcoordScale;
uniform vec2 rightTexcoordScale;
uniform vec2 viewportSize;
uniform bool flipLeftY = false;
uniform bool flipRightY = false;

const float MIN_BAR_WIDTH_PX = 4.0;
const float HANDLE_HEIGHT_PX = 80.0;
const float HANDLE_WIDTH_PX = 24.0;
const float GRIP_SPACING_PX = 10.0;
const float GRIP_WIDTH_PX = 2.0;
const float GRIP_LENGTH_PX = 12.0;
const float CORNER_RADIUS_PX = 6.0;
const int GRIP_LINE_COUNT = 2;

void main() {
    vec2 uv = TexCoord;

    float leftY = flipLeftY ? (1.0 - uv.y) : uv.y;
    float rightY = flipRightY ? (1.0 - uv.y) : uv.y;

    vec4 leftColor = texture(leftTexture, vec2(uv.x, leftY) * leftTexcoordScale);
    vec4 rightColor = texture(rightTexture, vec2(uv.x, rightY) * rightTexcoordScale);

    vec4 color = (uv.x < splitPosition) ? leftColor : rightColor;

    if (showDivider) {
        float distFromSplit = abs(uv.x - splitPosition);
        float barWidth = max(dividerWidth, MIN_BAR_WIDTH_PX / viewportSize.x);

        if (distFromSplit < barWidth * 0.5) {
            color = dividerColor;

            float handleHeight = HANDLE_HEIGHT_PX / viewportSize.y;
            float handleWidth = HANDLE_WIDTH_PX / viewportSize.x;
            float distFromCenter = abs(uv.y - 0.5);

            if (distFromCenter < handleHeight * 0.5 && distFromSplit < handleWidth * 0.5) {
                color = vec4(dividerColor.rgb * 0.8, 1.0);

                float gripSpacing = GRIP_SPACING_PX / viewportSize.y;
                float gripWidth = GRIP_WIDTH_PX / viewportSize.y;
                float gripLength = GRIP_LENGTH_PX / viewportSize.x;
                float localY = uv.y - 0.5;

                for (int i = -GRIP_LINE_COUNT; i <= GRIP_LINE_COUNT; i++) {
                    float lineY = float(i) * gripSpacing;
                    if (abs(localY - lineY) < gripWidth && distFromSplit < gripLength * 0.5) {
                        color = vec4(1.0, 1.0, 1.0, 0.9);
                    }
                }

                float cornerRadius = CORNER_RADIUS_PX / viewportSize.y;
                vec2 handleSize = vec2(handleWidth, handleHeight);
                vec2 localPos = vec2(distFromSplit, distFromCenter);
                vec2 cornerDist = localPos - (handleSize * 0.5 - cornerRadius);

                if (cornerDist.x > 0.0 && cornerDist.y > 0.0) {
                    if (length(cornerDist) > cornerRadius) {
                        color = dividerColor;
                    }
                }
            }
        }
    }

    FragColor = color;
}
