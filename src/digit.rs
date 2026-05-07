use cgmath::{Matrix4, Vector3};

use crate::graphics_manager::structures::{ModelMesh, Vertex};
use crate::graphics_manager::{GraphicsManager, MeshHandle, ModelHandle};

// Segment order: [top, top-left, top-right, middle, bottom-left, bottom-right, bottom].
#[rustfmt::skip]
const SEGMENTS_FOR_DIGIT: [[bool; 7]; 10] = [
    //   T      TL     TR     M      BL     BR     B
    [ true,  true,  true, false,  true,  true,  true], // 0
    [false, false,  true, false, false,  true, false], // 1
    [ true, false,  true,  true,  true, false,  true], // 2
    [ true, false,  true,  true, false,  true,  true], // 3
    [false,  true,  true,  true, false,  true, false], // 4
    [ true,  true, false,  true, false,  true,  true], // 5
    [ true,  true, false,  true,  true,  true,  true], // 6
    [ true, false,  true, false, false,  true, false], // 7
    [ true,  true,  true,  true,  true,  true,  true], // 8
    [ true,  true,  true,  true, false,  true,  true], // 9
];

const COLOR: [f32; 3] = [1.0, 1.0, 1.0];

// Far enough from the play field that perspective culls it.
const OFFSCREEN: Vector3<f32> = Vector3 {
    x: 1000.0,
    y: 1000.0,
    z: 0.0,
};

/// Two shared meshes — horizontal and vertical segments — registered once
/// and reused as instances by every digit. Saves 12 buffer allocations
/// (relative to the old per-segment registration) for two on-screen digits.
pub struct DigitMeshes {
    pub horizontal: MeshHandle,
    pub vertical: MeshHandle,
    pub segment_size: f32,
}

impl DigitMeshes {
    pub fn register(gm: &mut GraphicsManager, segment_size: f32) -> Self {
        let thickness = segment_size * 0.2;
        Self {
            horizontal: gm.register_mesh(&rect_mesh(segment_size, thickness)),
            vertical: gm.register_mesh(&rect_mesh(thickness, segment_size)),
            segment_size,
        }
    }
}

pub struct Digit {
    pub segments: [ModelHandle; 7],
    pub position: Vector3<f32>,
    pub segment_size: f32,
    pub value: u8,
}

impl Digit {
    pub fn new(gm: &mut GraphicsManager, meshes: &DigitMeshes, position: Vector3<f32>) -> Self {
        let h = meshes.horizontal;
        let v = meshes.vertical;
        let segments = [
            gm.register_instance(h, COLOR), // top
            gm.register_instance(v, COLOR), // top-left
            gm.register_instance(v, COLOR), // top-right
            gm.register_instance(h, COLOR), // middle
            gm.register_instance(v, COLOR), // bottom-left
            gm.register_instance(v, COLOR), // bottom-right
            gm.register_instance(h, COLOR), // bottom
        ];

        Self {
            segments,
            position,
            segment_size: meshes.segment_size,
            value: 0,
        }
    }

    pub fn set_value(&mut self, value: u8) {
        self.value = value;
    }

    pub fn get_model_transforms(&self) -> Vec<(ModelHandle, Matrix4<f32>)> {
        let lit = SEGMENTS_FOR_DIGIT[(self.value % 10) as usize];
        let s = self.segment_size;
        let half = s / 2.0;
        // Positive Y is downwards.
        let offsets = [
            Vector3 { x: 0.0,    y: -s,   z: 0.0 }, // top
            Vector3 { x: -half,  y: -half, z: 0.0 }, // top-left
            Vector3 { x:  half,  y: -half, z: 0.0 }, // top-right
            Vector3 { x: 0.0,    y: 0.0,  z: 0.0 }, // middle
            Vector3 { x: -half,  y:  half, z: 0.0 }, // bottom-left
            Vector3 { x:  half,  y:  half, z: 0.0 }, // bottom-right
            Vector3 { x: 0.0,    y:  s,   z: 0.0 }, // bottom
        ];

        (0..7)
            .map(|i| {
                let translation = if lit[i] {
                    self.position + offsets[i]
                } else {
                    OFFSCREEN
                };
                (self.segments[i], Matrix4::from_translation(translation))
            })
            .collect()
    }
}

fn rect_mesh(width: f32, height: f32) -> ModelMesh {
    let hw = width / 2.0;
    let hh = height / 2.0;
    ModelMesh {
        vertices: vec![
            Vertex { pos: [-hw, -hh] },
            Vertex { pos: [ hw, -hh] },
            Vertex { pos: [ hw,  hh] },
            Vertex { pos: [-hw,  hh] },
        ],
        indices: vec![0u32, 1, 2, 2, 3, 0],
    }
}
