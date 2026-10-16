"""Windows text clipboard: read (right-click paste) and write (right-click copy of a selection)."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes


def write_clipboard_text(text: str) -> None:
    if sys.platform != "win32":
        raise RuntimeError("Right-click copy is available on Windows; use Ctrl+C here.")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.CloseClipboard.restype = wintypes.BOOL
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]

    data = ctypes.create_unicode_buffer(text)
    size = ctypes.sizeof(data)
    handle = kernel32.GlobalAlloc(0x0002, size)  # GMEM_MOVEABLE
    if not handle:
        raise OSError("Could not allocate clipboard memory.")
    pointer = kernel32.GlobalLock(handle)
    if not pointer:
        kernel32.GlobalFree(handle)
        raise OSError("Could not write clipboard memory.")
    ctypes.memmove(pointer, data, size)
    kernel32.GlobalUnlock(handle)
    if not user32.OpenClipboard(None):
        kernel32.GlobalFree(handle)
        raise OSError("The clipboard is busy. Try again or use Ctrl+C.")
    try:
        user32.EmptyClipboard()
        if not user32.SetClipboardData(13, handle):  # CF_UNICODETEXT; the clipboard owns the memory now
            kernel32.GlobalFree(handle)
            raise OSError("Could not set the clipboard text.")
    finally:
        user32.CloseClipboard()


def read_clipboard_text() -> str:
    if sys.platform != "win32":
        raise RuntimeError("Right-click paste is available on Windows; use Ctrl+V here.")

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.CloseClipboard.restype = wintypes.BOOL
    kernel32.GlobalLock.argtypes = [wintypes.HANDLE]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HANDLE]

    if not user32.OpenClipboard(None):
        raise OSError("The clipboard is busy. Try Ctrl+V or Shift+Insert.")
    try:
        if user32.IsClipboardFormatAvailable(13):
            handle = user32.GetClipboardData(13)
            if not handle:
                raise OSError("Could not access clipboard text.")
            pointer = kernel32.GlobalLock(handle)
            if not pointer:
                raise OSError("Could not read clipboard text.")
            try:
                return ctypes.wstring_at(pointer)
            finally:
                kernel32.GlobalUnlock(handle)

        if user32.IsClipboardFormatAvailable(15):
            shell32 = ctypes.WinDLL("shell32", use_last_error=True)
            shell32.DragQueryFileW.argtypes = [wintypes.HANDLE, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
            shell32.DragQueryFileW.restype = wintypes.UINT
            handle = user32.GetClipboardData(15)
            if not handle:
                raise OSError("Could not access copied files.")
            count = shell32.DragQueryFileW(handle, 0xFFFFFFFF, None, 0)
            if not 1 <= count <= 50:
                raise ValueError("Select between 1 and 50 files to paste.")
            paths = []
            for index in range(count):
                length = shell32.DragQueryFileW(handle, index, None, 0)
                buffer = ctypes.create_unicode_buffer(length + 1)
                shell32.DragQueryFileW(handle, index, buffer, length + 1)
                paths.append(f'"{buffer.value}"')
            return "\n".join(paths)

        raise ValueError("The clipboard contains neither text nor copied files.")
    finally:
        user32.CloseClipboard()
