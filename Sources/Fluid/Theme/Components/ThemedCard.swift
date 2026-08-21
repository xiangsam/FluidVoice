import SwiftUI

enum ThemedCardStyle {
    case standard
    case prominent
    case subtle
}

struct ThemedCard<Content: View>: View {
    @Environment(\.theme) private var theme
    @State private var isHovered = false

    private let style: ThemedCardStyle
    private let hoverEffect: Bool
    private let padding: CGFloat?
    private let content: Content

    init(
        style: ThemedCardStyle = .standard,
        padding: CGFloat? = nil,
        hoverEffect: Bool = false,
        @ViewBuilder content: () -> Content
    ) {
        self.style = style
        self.padding = padding
        self.hoverEffect = hoverEffect
        self.content = content()
    }

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: 12, style: .continuous)

        self.content
            .padding(self.padding ?? 14)
            .background(
                shape
                    .fill(Color(nsColor: .controlBackgroundColor))
                    .overlay(
                        shape.stroke(
                            self.isHovered && self.hoverEffect ? self.theme.palette.accent.opacity(0.45) : Color.secondary.opacity(0.18),
                            lineWidth: 1
                        )
                    )
            )
            .contentShape(shape)
            .onHover { hovering in
                guard self.hoverEffect else { return }
                self.isHovered = hovering
            }
            .animation(.easeOut(duration: 0.15), value: self.isHovered)
    }
}

// MARK: - Configuration

private extension ThemedCard {
    struct CardConfiguration {
        let background: Color
        let border: Color
        let borderOpacity: Double
        let hoverBorderOpacity: Double
        let borderWidth: CGFloat
        let material: Material
        let cornerRadius: CGFloat
        let shadow: AppTheme.Metrics.Shadow
        let hoverShadowBoost: Double

        init(style: ThemedCardStyle, theme: AppTheme) {
            let cardSurface = theme.metrics.cardSurface
            let variant: AppTheme.Metrics.CardSurface.Variant

            switch style {
            case .standard:
                variant = cardSurface.standard
                self.background = theme.palette.cardBackground
                self.border = theme.palette.cardBorder
                self.material = theme.materials.card
                self.cornerRadius = theme.metrics.corners.lg
                self.shadow = theme.metrics.cardShadow
            case .prominent:
                variant = cardSurface.prominent
                self.background = theme.palette.elevatedCardBackground
                self.border = theme.palette.accent
                self.material = theme.materials.elevatedCard
                self.cornerRadius = theme.metrics.corners.lg
                self.shadow = theme.metrics.elevatedCardShadow
            case .subtle:
                variant = cardSurface.subtle
                self.background = theme.palette.contentBackground
                self.border = theme.palette.cardBorder
                self.material = theme.materials.card
                self.cornerRadius = theme.metrics.corners.md
                self.shadow = theme.metrics.cardShadow
            }

            self.borderOpacity = variant.borderOpacity
            self.hoverBorderOpacity = variant.hoverBorderOpacity
            self.borderWidth = variant.borderWidth
            self.hoverShadowBoost = variant.hoverShadowBoost
        }
    }
}
