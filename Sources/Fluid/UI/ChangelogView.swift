import SwiftUI

struct ChangelogView: View {
    @Environment(\.theme) private var theme

    var body: some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(alignment: .leading, spacing: self.theme.metrics.spacing.xl) {
                HStack(spacing: 8) {
                    Image(systemName: "sparkles.rectangle.stack")
                        .font(.title2)
                        .foregroundStyle(self.theme.palette.accent)
                    Text("Changelog".loc)
                        .font(.title2)
                        .fontWeight(.bold)
                }

                ThemedCard(style: .standard, hoverEffect: false) {
                    VStack(alignment: .leading, spacing: self.theme.metrics.spacing.md) {
                        Text("更新日志已随应用自动更新停用;发行说明请查看 GitHub Releases 页面。".loc)
                            .font(self.theme.typography.body)
                            .foregroundStyle(self.theme.palette.primaryText)
                        Button {
                            if let url = URL(string: "https://github.com/xiangsam/MlxVoice/releases") {
                                NSWorkspace.shared.open(url)
                            }
                        } label: {
                            Label("Open GitHub Releases".loc, systemImage: "arrow.up.right.square")
                        }
                        .buttonStyle(.bordered)
                    }
                    .padding()
                }
            }
            .padding()
        }
    }
}
