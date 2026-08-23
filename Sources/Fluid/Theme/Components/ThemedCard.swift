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
