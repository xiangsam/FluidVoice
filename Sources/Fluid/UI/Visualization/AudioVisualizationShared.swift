import Combine
import SwiftUI

final class AudioVisualizationData: ObservableObject {
    @Published var audioLevel: CGFloat = 0.0
    private var cancellable: AnyCancellable?

    init(audioLevelPublisher: AnyPublisher<CGFloat, Never>) {
        self.cancellable = audioLevelPublisher
            .receive(on: RunLoop.main)
            .sink { [weak self] level in
                self?.audioLevel = level
            }
    }

    deinit {
        cancellable?.cancel()
    }
}
