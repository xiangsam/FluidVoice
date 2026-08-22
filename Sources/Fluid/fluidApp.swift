//
//  fluidApp.swift
//  fluid
//
//  Created by Barathwaj Anandan on 7/30/25.
//

import AppKit
import ApplicationServices
import SwiftUI

@main
struct FluidApp: App {
    @StateObject private var menuBarManager = MenuBarManager()
    @StateObject private var appServices: AppServices
    @ObservedObject private var settings = SettingsStore.shared
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    init() {
        // Use the shared singleton instance
        _appServices = StateObject(wrappedValue: AppServices.shared)
    }

    var body: some Scene {
        WindowGroup(id: "main") {
            AdaptiveAppTheme(accent: self.settings.accentColor) {
                ContentView()
                    .environmentObject(self.menuBarManager)
                    .environmentObject(self.appServices)
                    .background(
                        WindowAccessor { window in
                            window.title = "FluidVoice"
                            window.identifier = NSUserInterfaceItemIdentifier("FluidVoice.MainWindow")
                            window.isReleasedWhenClosed = false

                            // Ensure launch window uses the comfortable 1000x700 size centered on screen
                            if window.frame.width < 900 || window.frame.height < 600 {
                                let defaultSize = NSSize(width: 1000, height: 700)
                                let screenFrame = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: defaultSize.width, height: defaultSize.height)
                                let targetFrame = NSRect(
                                    x: screenFrame.midX - defaultSize.width / 2,
                                    y: screenFrame.midY - defaultSize.height / 2,
                                    width: defaultSize.width,
                                    height: defaultSize.height
                                )
                                window.setFrame(targetFrame, display: true, animate: false)
                            }
                        }
                    )
            }
        }
        .defaultSize(width: 1000, height: 700)
        .windowStyle(.hiddenTitleBar)
        .commands {
            CommandGroup(replacing: .appSettings) {
                Button("Settings...") {
                    self.menuBarManager.openPreferencesFromUI()
                }
                .keyboardShortcut(",", modifiers: .command)
            }
        }
    }
}
